"""
龙虾大脑（Agent Brain）v3 - 角色适配版

v3 核心改进:
  1. 角色识别: 根据 player.character 自动适配策略
  2. 出牌节奏: 增大所有延迟，最后一张牌到结束回合之间有 2.5 秒缓冲
  3. 选卡策略: 根据角色被动技能评分选牌，防止牌组膨胀
  4. 地图/篝火: 根据角色特性调整 HP 阈值
  5. 目标: 通关第一章 BOSS
"""

import sys
import os
import time
import json
import logging
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcp_connector
import combat_analyzer
import character_strategy

# ============================================================
# 全局配置
# ============================================================
ACTION_DELAY = 2.0       # 每次出牌后的等待秒数（等动画）
PRE_END_DELAY = 2.5      # 最后一张牌打完到结束回合的等待
END_TURN_DELAY = 4.0     # 结束回合后的等待秒数（等敌方行动动画）
UI_DELAY = 2.5           # 界面切换后的等待
POLL_INTERVAL = 2.0      # 主循环轮询间隔

# 角色档案（首次识别后赋值）
_current_profile = None
_current_character = None
_current_fight_type = "monster"  # "monster", "elite", "boss" - 决定是否用药水

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger("sts2_brain")


# ============================================================
# LLM 接口（预留）
# ============================================================
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """预留 LLM 挂载点，当前返回 None 使用规则兜底"""
    return None


def parse_llm_actions(llm_response: str) -> list:
    if not llm_response:
        return []
    try:
        match = re.search(r'\[.*\]', llm_response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return []


# ============================================================
# 实时出牌辅助（解决索引漂移问题）
# ============================================================
def find_card_in_hand(hand: list, card_name: str, exclude_indices: set = None) -> dict:
    """在当前手牌中按名称查找一张牌"""
    exclude = exclude_indices or set()
    for card in hand:
        if card.get("name") == card_name and card.get("index") not in exclude and card.get("can_play", False):
            return card
    return None


def play_card_realtime(card_name: str, target_id: str = None) -> bool:
    """实时出一张牌：先获取最新状态找牌，再出牌
    
    Returns:
        True=成功, False=失败
    """
    state = mcp_connector.get_game_state()
    if "error" in state:
        return False
    if state.get("state_type") not in ("monster", "elite", "boss"):
        logger.warning(f"不在战斗状态了 (当前: {state.get('state_type')}), 停止出牌")
        return False
    
    hand = state.get("player", {}).get("hand", [])
    card = find_card_in_hand(hand, card_name)
    
    if not card:
        logger.warning(f"手牌中找不到 '{card_name}'")
        return False
    
    # 确定 target
    target = None
    if card.get("target_type") == "AnyEnemy":
        # 如果指定了目标用指定的，否则选最低血量敌人
        if target_id:
            target = target_id
        else:
            enemies = state.get("battle", {}).get("enemies", [])
            alive = [e for e in enemies if e.get("hp", 0) > 0]
            if alive:
                target = min(alive, key=lambda e: e.get("hp", 999)).get("entity_id")
    
    result = mcp_connector.combat_play_card(card["index"], target)
    if "error" in result:
        logger.warning(f"出牌失败 [{card_name}]: {result.get('error')}")
        return False
    else:
        logger.info(f"🃏 出牌: {card_name} (索引{card['index']}) → {target or '无目标'}  {result.get('message', '')}")
        time.sleep(ACTION_DELAY)
        return True


# ============================================================
# 战斗决策引擎 v2（优化出牌顺序）
# ============================================================
def make_combat_decisions(game_state: dict) -> list:
    """生成本回合的出牌计划
    
    出牌优先级（像人一样思考）:
    1. 危急时使用药水
    2. 先打能力牌（长期增益）
    3. 再打 Debuff 牌（如 Bash 上易伤，让后续攻击增伤 50%）
    4. 检查斩杀：能秒就秒杀（最好的防守）
    5. 用剩余费打攻击牌
    6. 用剩余费打防御牌
    7. 结束回合
    
    Returns:
        [{"type": "play", "name": "Bash", "target": "jaw_worm_0"},
         {"type": "play", "name": "Strike", "target": "jaw_worm_0"},
         {"type": "play", "name": "Defend"},
         {"type": "potion", "slot": 0, "target": None},
         {"type": "end_turn"}]
    """
    player = game_state.get("player", {})
    battle = game_state.get("battle", {})
    
    if not battle.get("is_play_phase", False):
        return []
    
    hand = player.get("hand", [])
    energy = player.get("energy", 0)
    enemies = battle.get("enemies", [])
    
    if not hand or energy <= 0:
        return [{"type": "end_turn"}]
    
    threat_info = combat_analyzer.analyze_threats(game_state)
    hand_info = combat_analyzer.analyze_hand(game_state)
    potion_info = combat_analyzer.analyze_potions(game_state)
    
    plan = []
    remaining_energy = energy
    used_indices = set()  # 用索引追踪！不能用名字（同名牌会被跳过）
    
    # 选出主目标（血量最低的敌人）
    alive_enemies = [e for e in enemies if e.get("hp", 0) > 0]
    main_target = min(alive_enemies, key=lambda e: e.get("hp", 999)) if alive_enemies else None
    main_target_id = main_target.get("entity_id") if main_target else None
    
    # ===== 步骤 0: 药水使用（只在精英/Boss使用，小怪不浪费）=====
    total_incoming = threat_info["total_incoming_damage"]
    current_block = player.get("block", 0)
    current_hp = player.get("hp", 0)
    net_damage = max(0, total_incoming - current_block)
    is_hard_fight = _current_fight_type in ("elite", "boss")
    
    combat_potions = [p for p in potion_info["potions"] if p.get("can_use_in_combat", False)]
    
    # 0a) 致命威胁 → 任何战斗都用药保命
    if net_damage >= current_hp and net_damage > 0 and combat_potions:
        logger.info(f"!! 致命威胁！净伤害{net_damage} >= HP{current_hp}，使用药水保命...")
        for p in combat_potions:
            target = main_target_id if p.get("target_type") == "AnyEnemy" else None
            plan.append({"type": "potion", "slot": p["slot"], "target": target, "name": p.get("name", "?")})
            logger.info(f"  -> 药水: {p.get('name', '?')} (slot {p['slot']})")
            break
    
    # 0b) 精英/Boss战 + HP < 60% → 使用回血药
    elif is_hard_fight and current_hp < player.get("max_hp", 1) * 0.6:
        for p in combat_potions:
            combined = f"{p.get('description', '')} {p.get('name', '')} {p.get('id', '')}".lower()
            if any(kw in combined for kw in ["heal", "hp", "fairy", "blood", "回复", "治疗"]):
                target = main_target_id if p.get("target_type") == "AnyEnemy" else None
                plan.append({"type": "potion", "slot": p["slot"], "target": target, "name": p.get("name", "?")})
                logger.info(f"  -> {_current_fight_type}战，HP低，使用回血药水: {p.get('name', '?')}")
                break
    
    # 0c) 精英/Boss战回合1 → 使用增益药水（力量/敏捷等）
    elif is_hard_fight and battle.get("round", 0) <= 1:
        for p in combat_potions:
            combined = f"{p.get('description', '')} {p.get('name', '')} {p.get('id', '')}".lower()
            # 增益类药水在精英/Boss开局使用
            if any(kw in combined for kw in ["strength", "dexterity", "力量", "敏捷", "focus", "集中"]):
                target = main_target_id if p.get("target_type") == "AnyEnemy" else None
                plan.append({"type": "potion", "slot": p["slot"], "target": target, "name": p.get("name", "?")})
                logger.info(f"  -> {_current_fight_type}开局，使用增益药水: {p.get('name', '?')}")
                break
    
    # ===== 步骤 1: 打能力牌（长期增益优先）=====
    for card in hand_info["power_cards"]:
        if card["index"] in used_indices or not card["can_play"] or card["cost"] > remaining_energy:
            continue
        plan.append({"type": "play", "name": card["name"], "index": card["index"]})
        remaining_energy -= card["cost"]
        used_indices.add(card["index"])
        logger.info(f"  [能力] {card['name']} (费{card['cost']})")
    
    # ===== 步骤 2: 打 Debuff 牌（先上易伤/虚弱，让后续攻击更痛）=====
    for card in hand_info["debuff_cards"]:
        if card["index"] in used_indices or not card["can_play"] or card["cost"] > remaining_energy:
            continue
        plan.append({"type": "play", "name": card["name"], "target": main_target_id, "index": card["index"]})
        remaining_energy -= card["cost"]
        used_indices.add(card["index"])
        logger.info(f"  [Debuff] {card['name']} (费{card['cost']}) -> 先上debuff再输出")
    
    # ===== 步骤 3: 用剩余费用打攻击牌 =====
    for card in hand_info["attack_cards"]:
        if card["index"] in used_indices or not card["can_play"] or card["cost"] > remaining_energy:
            continue
        plan.append({"type": "play", "name": card["name"], "target": main_target_id, "index": card["index"]})
        remaining_energy -= card["cost"]
        used_indices.add(card["index"])
        logger.info(f"  [攻击] {card['name']} (费{card['cost']})")
    
    # ===== 步骤 4: 用剩余费用防御 =====
    for card in hand_info["block_cards"]:
        if card["index"] in used_indices or not card["can_play"] or card["cost"] > remaining_energy:
            continue
        plan.append({"type": "play", "name": card["name"], "index": card["index"]})
        remaining_energy -= card["cost"]
        used_indices.add(card["index"])
        logger.info(f"  [防御] {card['name']} (费{card['cost']})")
    
    # ===== 步骤 5: 其他技能牌（有费用就打，别浪费）=====
    for card in hand_info["other_cards"]:
        if card["index"] in used_indices or not card["can_play"] or card["cost"] > remaining_energy:
            continue
        plan.append({"type": "play", "name": card["name"], "index": card["index"]})
        remaining_energy -= card["cost"]
        used_indices.add(card["index"])
        logger.info(f"  [其他] {card['name']} (费{card['cost']})")
    
    if remaining_energy > 0:
        logger.info(f"  ⚠ 剩余 {remaining_energy} 费用无法使用（无可出的牌）")
    
    plan.append({"type": "end_turn"})
    return plan


# ============================================================
# 战斗执行器
# ============================================================
def handle_combat(game_state: dict):
    """处理一个完整的战斗回合"""
    global _current_profile, _current_character, _current_fight_type
    state_type = game_state.get("state_type", "")
    _current_fight_type = state_type  # "monster", "elite", "boss"
    battle = game_state.get("battle", {})
    player = game_state.get("player", {})
    
    # 首次识别角色
    char_name = player.get("character", "")
    if char_name and char_name != _current_character:
        _current_character = char_name
        _current_profile = character_strategy.get_character_profile(char_name)
        logger.info(f"=== 识别角色: {char_name} ({_current_profile['name_cn']}) ===")
        logger.info(f"   被动: {_current_profile['passive']}")
        logger.info(f"   策略: {_current_profile['strategy']}")
    
    # 如果不是玩家回合，等待
    if battle.get("turn") != "player" or not battle.get("is_play_phase", False):
        logger.info(f"等待玩家回合... (当前: {battle.get('turn')}, 出牌阶段: {battle.get('is_play_phase')})")
        time.sleep(2.0)
        return
    
    logger.info(f"=== 战斗回合 {battle.get('round', '?')} ({state_type}) ===")
    
    # 生成战场简报
    briefing = combat_analyzer.generate_combat_briefing(game_state)
    logger.info("\n" + briefing)
    
    # 生成决策计划
    logger.info("生成出牌计划...")
    plan = make_combat_decisions(game_state)
    logger.info(f"   计划包含 {len(plan)} 个动作")
    
    # 逐步执行
    for i, action in enumerate(plan):
        act_type = action.get("type")
        
        if act_type == "potion":
            result = mcp_connector.use_potion(action["slot"], action.get("target"))
            if "error" in result:
                logger.warning(f"药水使用失败: {result.get('error')}")
            else:
                logger.info(f"🧪 使用药水: {action.get('name', '?')}  {result.get('message', '')}")
            time.sleep(ACTION_DELAY)
        
        elif act_type == "play":
            success = play_card_realtime(action["name"], action.get("target"))
            if not success:
                logger.info(f"  跳过 {action['name']}（找不到或无法出牌）")
        
        elif act_type == "end_turn":
            # 最后一张牌打完后，等一会再结束（解决"太快"问题）
            logger.info(f"   等待 {PRE_END_DELAY} 秒后结束回合...")
            time.sleep(PRE_END_DELAY)
            result = mcp_connector.combat_end_turn()
            if "error" in result:
                logger.warning(f"结束回合失败: {result.get('error')}")
            else:
                logger.info(f"   回合结束，等待敌方行动...")
            time.sleep(END_TURN_DELAY)


def handle_hand_select(game_state: dict):
    """处理战斗中选牌（丢弃/消耗提示）"""
    hand_select = game_state.get("hand_select", {})
    logger.info(f"📝 需要选牌: {hand_select}")
    cards = hand_select.get("cards", [])
    if cards:
        # 选最后一张（一般是最不重要的）
        idx = len(cards) - 1
        mcp_connector.combat_select_card(idx)
        time.sleep(UI_DELAY)
        mcp_connector.combat_confirm_selection()
        time.sleep(UI_DELAY)


def handle_map(game_state: dict):
    """处理地图选路 - 智能评分系统"""
    global _current_profile
    map_data = game_state.get("map", {})
    next_options = map_data.get("next_options", [])
    player = game_state.get("player", {})
    current_hp = player.get("hp", 0)
    max_hp = max(player.get("max_hp", 1), 1)
    hp_ratio = current_hp / max_hp
    
    if not next_options:
        logger.info("地图上没有可选节点")
        return
    
    profile = _current_profile or character_strategy.DEFAULT_PROFILE
    elite_threshold = profile.get("elite_hp_threshold", 0.7)
    
    # 给每个路线打分
    def score_node(opt):
        """根据当前HP状态给路线打分"""
        node_type = opt.get("type", "").lower()
        s = 0
        
        # --- 根据节点类型 + HP状况打分 ---
        if "rest" in node_type:
            if hp_ratio < 0.4:
                s = 100   # 血少时篝火最优先
            elif hp_ratio < 0.6:
                s = 60
            else:
                s = 20    # 血多时篝火价值低（但升级还是有价值）
        
        elif "elite" in node_type:
            if hp_ratio > elite_threshold:
                s = 80    # 血多打精英拿遗物
            elif hp_ratio > 0.5:
                s = 30    # 半血打精英有风险
            else:
                s = -50   # 血少别碰精英
        
        elif "monster" in node_type:
            if hp_ratio > 0.5:
                s = 50    # 正常刷怪获取卡牌
            elif hp_ratio > 0.3:
                s = 25
            else:
                s = 5     # 血太少了尽量避免
        
        elif "event" in node_type or "?" in node_type:
            s = 45        # 事件有随机性，但平均收益不错
            if hp_ratio < 0.4:
                s = 40    # 血少时事件可能给回血
        
        elif "merchant" in node_type or "shop" in node_type:
            s = 35        # 商店买关键牌/移除垃圾牌
        
        elif "treasure" in node_type:
            s = 55        # 宝箱白嫖遗物
        
        elif "boss" in node_type:
            s = 10        # Boss必须打，但不额外加分
        
        # --- 前瞻加分: 看这条路后面通向哪里 ---
        leads_to = opt.get("leads_to", [])
        for next_node in leads_to:
            next_type = next_node.get("type", "").lower()
            if "rest" in next_type and hp_ratio < 0.5:
                s += 10   # 后面有篝火，可以先打怪再回血
            elif "elite" in next_type and hp_ratio > 0.6:
                s += 5
        
        return s
    
    # 给所有选项打分
    scored = []
    for opt in next_options:
        s = score_node(opt)
        node_type = opt.get("type", "?")
        scored.append((s, opt, node_type))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # 打印所有选项的分数
    for s, opt, nt in scored:
        logger.info(f"  路线: {nt} = {s}分")
    
    best_score, choice, best_type = scored[0]
    logger.info(f"   选择路线: {best_type} ({best_score}分, HP:{hp_ratio:.0%}) 索引={choice.get('index')}")
    mcp_connector.map_choose_node(choice["index"])
    time.sleep(UI_DELAY)


def handle_rewards(game_state: dict):
    """处理战斗奖励 - 循环领取所有奖励"""
    max_claims = 10  # 防无限循环
    for attempt in range(max_claims):
        # 每次操作后重新获取状态（领取后索引会变）
        if attempt > 0:
            time.sleep(UI_DELAY)
            game_state = mcp_connector.get_game_state()
            if "error" in game_state:
                break
        
        state_type = game_state.get("state_type", "")
        
        # 如果已经不在奖励界面了
        if state_type == "card_reward":
            handle_card_reward(game_state)
            return
        elif state_type == "map":
            return
        elif state_type != "rewards":
            time.sleep(UI_DELAY)
            return
        
        rewards = game_state.get("rewards", {})
        items = rewards.get("items", [])
        can_proceed = rewards.get("can_proceed", False)
        
        if not items:
            # 没有奖励了，返回地图
            if can_proceed:
                logger.info("   奖励全部领完，返回地图")
                mcp_connector.proceed_to_map()
                time.sleep(UI_DELAY)
            return
        
        # 按优先级领取: 金币 > 遗物 > 药水 > 卡牌
        claimed = False
        for item in items:
            item_type = item.get("type", "").lower()
            desc = item.get("description", item_type)
            idx = item.get("index", 0)
            
            if item_type in ("gold", "relic"):
                logger.info(f"   领取[{item_type}]: {desc}")
                result = mcp_connector.rewards_claim(idx)
                if "error" in result:
                    logger.warning(f"   领取失败: {result.get('error')}")
                else:
                    logger.info(f"   -> {result.get('message', 'OK')}")
                claimed = True
                break  # 重新获取状态
            
            elif item_type == "potion":
                logger.info(f"   领取[药水]: {desc}")
                result = mcp_connector.rewards_claim(idx)
                if "error" in result:
                    # 药水槽满了，跳过这个奖励
                    logger.warning(f"   药水领取失败(可能槽满): {result.get('error')}")
                    continue  # 尝试下一个奖励
                else:
                    logger.info(f"   -> {result.get('message', 'OK')}")
                claimed = True
                break
            
            elif item_type in ("card", "special_card"):
                logger.info(f"   领取[卡牌奖励]: {desc}")
                result = mcp_connector.rewards_claim(idx)
                if "error" in result:
                    logger.warning(f"   卡牌领取失败: {result.get('error')}")
                    continue
                # claim_reward 会打开选卡界面
                time.sleep(UI_DELAY)
                new_state = mcp_connector.get_game_state()
                if new_state.get("state_type") == "card_reward":
                    handle_card_reward(new_state)
                claimed = True
                break
        
        if not claimed:
            # 所有剩余奖励都无法领取，直接返回地图
            if can_proceed:
                logger.info("   剩余奖励无法领取，返回地图")
                mcp_connector.proceed_to_map()
                time.sleep(UI_DELAY)
            return


def handle_card_reward(game_state: dict):
    """处理卡牌奖励 - 根据角色策略 + 牌组需求评分选牌"""
    global _current_profile
    card_reward = game_state.get("card_reward", {})
    cards = card_reward.get("cards", [])
    
    if not cards:
        logger.info("没有可选卡牌，跳过")
        mcp_connector.rewards_skip_card()
        time.sleep(UI_DELAY)
        return
    
    profile = _current_profile or character_strategy.DEFAULT_PROFILE
    player = game_state.get("player", {})
    card_names = [c.get("name", "?") for c in cards]
    logger.info(f"   卡牌奖励: {card_names}")
    
    # 分析当前牌组构成
    all_deck_cards = []
    all_deck_cards.extend(player.get("draw_pile", []))
    all_deck_cards.extend(player.get("discard_pile", []))
    all_deck_cards.extend(player.get("hand", []))
    deck_needs = character_strategy.analyze_deck_needs(all_deck_cards)
    
    # 打印牌组分析
    logger.info(f"  牌组分析: {deck_needs['total']}张 "
                f"(攻{deck_needs['attack_count']} 防{deck_needs['block_count']} "
                f"能{deck_needs['power_count']} 技{deck_needs['skill_count']})")
    if deck_needs["need_attack"]:
        logger.info("  -> 缺攻击牌！")
    if deck_needs["need_block"]:
        logger.info("  -> 缺防御牌！")
    if deck_needs["need_power"]:
        logger.info("  -> 缺能力牌！")
    
    # 根据角色策略 + 牌组需求给每张牌打分
    scores = []
    for i, card in enumerate(cards):
        score = character_strategy.score_card_for_character(card, profile, deck_needs)
        scores.append((i, score, card.get('name', '?'), card.get('type', '?')))
        logger.info(f"  评分: {card.get('name', '?')} [{card.get('type', '?')}] = {score:.1f}分")
    
    scores.sort(key=lambda x: x[1], reverse=True)
    best_idx, best_score, best_name, best_type = scores[0]
    
    # 检查是否应该跳过（防止牌组膨胀）
    deck_size_est = player.get("draw_pile_count", 5) + player.get("discard_pile_count", 0) + len(player.get("hand", [])) + player.get("exhaust_pile_count", 0)
    
    if character_strategy.should_skip_card_reward(cards, profile, deck_size_est, deck_needs):
        logger.info(f"  -> 牌组已有约{deck_size_est}张，最高分仅{best_score:.1f}，跳过防膨胀")
        mcp_connector.rewards_skip_card()
    else:
        logger.info(f"  -> 选择: {best_name} ({best_score:.1f}分)")
        mcp_connector.rewards_pick_card(best_idx)
    
    time.sleep(UI_DELAY)
    
    # 看看回到了哪个界面
    new_state = mcp_connector.get_game_state()
    if new_state.get("state_type") == "rewards":
        can_proceed = new_state.get("rewards", {}).get("can_proceed", False)
        if can_proceed:
            mcp_connector.proceed_to_map()
            time.sleep(UI_DELAY)


def handle_rest_site(game_state: dict):
    """处理篝火 - 根据角色策略调整休息阈值"""
    global _current_profile
    rest = game_state.get("rest_site", {})
    options = rest.get("options", [])
    player = game_state.get("player", {})
    hp_ratio = player.get("hp", 0) / max(player.get("max_hp", 1), 1)
    
    if not options:
        logger.info("篝火无可用选项")
        return
    
    profile = _current_profile or character_strategy.DEFAULT_PROFILE
    rest_threshold = profile.get("hp_rest_threshold", 0.55)
    
    chosen = None
    for opt in options:
        if not opt.get("is_enabled", True):
            continue
        opt_id = opt.get("id", "").lower()
        opt_name = opt.get("name", "").lower()
        
        if hp_ratio < rest_threshold and ("rest" in opt_id or "rest" in opt_name):
            chosen = opt
            break
        elif hp_ratio >= rest_threshold and ("smith" in opt_id or "upgrade" in opt_id or "smith" in opt_name):
            chosen = opt
            break
    
    if not chosen:
        chosen = options[0]
    
    logger.info(f"   篝火: {chosen.get('name', '?')} (HP:{hp_ratio:.0%}, 阈值:{rest_threshold:.0%})")
    mcp_connector.rest_choose_option(chosen["index"])
    time.sleep(UI_DELAY * 2)  # 篝火动画长一些
    
    # 可能弹出升级选卡
    new_state = mcp_connector.get_game_state()
    if new_state.get("state_type") == "card_select":
        handle_card_select(new_state)
    elif new_state.get("state_type") not in ("map", "rest_site"):
        time.sleep(UI_DELAY)
        new_state2 = mcp_connector.get_game_state()
        if new_state2.get("state_type") == "card_select":
            handle_card_select(new_state2)


def handle_event(game_state: dict):
    """处理事件"""
    event = game_state.get("event", {})
    options = event.get("options", [])
    in_dialogue = event.get("in_dialogue", False)
    
    if in_dialogue:
        logger.info("💬 推进对话...")
        mcp_connector.event_advance_dialogue()
        time.sleep(UI_DELAY)
        return
    
    if not options:
        return
    
    # 过滤可选项
    available = [o for o in options if not o.get("is_locked") and not o.get("was_chosen")]
    
    if not available:
        proceed = [o for o in options if o.get("is_proceed")]
        if proceed:
            mcp_connector.event_choose_option(proceed[0]["index"])
            time.sleep(UI_DELAY)
        return
    
    choice = available[0]
    logger.info(f"📜 事件选择: {choice.get('title', '?')} — {choice.get('description', '')[:50]}")
    mcp_connector.event_choose_option(choice["index"])
    time.sleep(UI_DELAY)


def handle_shop(game_state: dict):
    """处理商店"""
    logger.info("🏪 商店界面，暂时跳过...")
    mcp_connector.proceed_to_map()
    time.sleep(UI_DELAY)


def handle_card_select(game_state: dict):
    """处理卡牌选择（升级/变换/移除）"""
    card_select = game_state.get("card_select", {})
    cards = card_select.get("cards", [])
    screen_type = card_select.get("screen_type", "unknown")
    
    logger.info(f"📋 卡牌选择界面 ({screen_type})，共 {len(cards)} 张牌")
    
    if cards:
        # 升级时优先选攻击牌
        best_idx = 0
        for i, card in enumerate(cards):
            card_type = card.get("type", "").lower()
            if card_type == "attack":
                best_idx = i
                break
        
        logger.info(f"  → 选择: {cards[best_idx].get('name', '?')}")
        mcp_connector.deck_select_card(best_idx)
        time.sleep(UI_DELAY)
        
        # 等待预览后确认
        new_state = mcp_connector.get_game_state()
        if new_state.get("state_type") == "card_select":
            can_confirm = new_state.get("card_select", {}).get("can_confirm", False)
            if can_confirm:
                mcp_connector.deck_confirm_selection()
                time.sleep(UI_DELAY)


def handle_treasure(game_state: dict):
    """处理宝箱"""
    treasure = game_state.get("treasure", {})
    relics = treasure.get("relics", [])
    
    for i, relic in enumerate(relics):
        logger.info(f"💎 领取宝箱遗物: {relic.get('name', '?')}")
        mcp_connector.treasure_claim_relic(i)
        time.sleep(UI_DELAY)
    
    mcp_connector.proceed_to_map()
    time.sleep(UI_DELAY)


def handle_relic_select(game_state: dict):
    """处理遗物选择（如 BOSS 遗物三选一）"""
    relic_select = game_state.get("relic_select", {})
    relics = relic_select.get("relics", [])
    
    if relics:
        logger.info(f"🔮 遗物选择: {[r.get('name', '?') for r in relics]}")
        mcp_connector.relic_select(0)
        time.sleep(UI_DELAY)
    else:
        mcp_connector.relic_skip()
        time.sleep(UI_DELAY)


def handle_bundle_select(game_state: dict):
    """处理捆绑包选择"""
    logger.info("📦 捆绑包选择，选第一个...")
    mcp_connector.bundle_select(0)
    time.sleep(UI_DELAY)
    mcp_connector.bundle_confirm_selection()
    time.sleep(UI_DELAY)


# ============================================================
# 需要额外导入的 connector 方法
# ============================================================
def _ensure_connector_methods():
    """确保 connector 有 bundle 相关方法"""
    if not hasattr(mcp_connector, 'bundle_select'):
        def bundle_select(idx):
            return mcp_connector._post({"action": "select_bundle", "index": idx})
        mcp_connector.bundle_select = bundle_select
    if not hasattr(mcp_connector, 'bundle_confirm_selection'):
        def bundle_confirm():
            return mcp_connector._post({"action": "confirm_bundle_selection"})
        mcp_connector.bundle_confirm_selection = bundle_confirm

_ensure_connector_methods()


# ============================================================
# 主循环
# ============================================================
STATE_HANDLERS = {
    "monster": handle_combat,
    "elite": handle_combat,
    "boss": handle_combat,
    "hand_select": handle_hand_select,
    "map": handle_map,
    "rewards": handle_rewards,
    "card_reward": handle_card_reward,
    "rest_site": handle_rest_site,
    "event": handle_event,
    "shop": handle_shop,
    "fake_merchant": handle_shop,
    "card_select": handle_card_select,
    "treasure": handle_treasure,
    "relic_select": handle_relic_select,
    "bundle_select": handle_bundle_select,
}


def main_loop(max_retries: int = 999):
    """龙虾全自动游戏主循环"""
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║  🦞 龙虾 (OpenClaw) v2 全自动模式    ║")
    logger.info("╚══════════════════════════════════════╝")
    logger.info("等待游戏连接... 请确保:")
    logger.info("  1. 杀戮尖塔2 已启动")
    logger.info("  2. 游戏设置中已启用 STS2_MCP Mod")
    logger.info("  3. 已开始一局游戏")
    logger.info("脚本会持续等待，直到连接成功。按 Ctrl+C 可退出。")
    
    consecutive_errors = 0
    last_state_type = None
    same_state_count = 0
    connected = False
    
    while True:
        try:
            state = mcp_connector.get_game_state()
            
            if "error" in state:
                consecutive_errors += 1
                if consecutive_errors <= 3 or consecutive_errors % 10 == 0:
                    logger.info(f"⏳ 等待游戏连接... ({consecutive_errors}次)")
                if consecutive_errors >= max_retries:
                    logger.error("连续错误次数过多，退出。")
                    break
                time.sleep(5.0)
                continue
            
            # 首次连接成功
            if not connected:
                connected = True
                logger.info("✅ 游戏连接成功！龙虾开始工作！")
            
            consecutive_errors = 0
            state_type = state.get("state_type", "unknown")
            
            # 防卡死检测
            if state_type == last_state_type:
                same_state_count += 1
                if same_state_count > 15:
                    logger.warning(f"⚠️ 连续 {same_state_count} 次处于 {state_type} 状态，可能卡住了。重试...")
                    time.sleep(5.0)
                    same_state_count = 0
                    continue
            else:
                same_state_count = 0
                last_state_type = state_type
            
            logger.info(f"━━━ 当前界面: {state_type} ━━━")
            
            # 游戏结束
            if state_type == "menu":
                logger.info("🏁 游戏结束（回到主菜单）。龙虾收工！")
                break
            
            # 路由
            handler = STATE_HANDLERS.get(state_type)
            if handler:
                handler(state)
            else:
                logger.info(f"⏳ 未处理界面: {state_type}，等待...")
                time.sleep(3.0)
            
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("\n👋 用户中断，龙虾下班了。")
            break
        except Exception as e:
            logger.exception(f"❌ 未预期异常: {e}")
            time.sleep(3.0)


if __name__ == "__main__":
    main_loop()
