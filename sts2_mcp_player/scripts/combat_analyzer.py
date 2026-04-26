"""
战斗分析器 - 基于 STS2MCP 真实数据结构重写 (v2: 增加 debuff 分类)

真实数据路径:
  - 手牌: state["player"]["hand"]  (列表, 每项含 index/name/type/cost/can_play/target_type)
  - 费用: state["player"]["energy"] / state["player"]["max_energy"]
  - 玩家格挡: state["player"]["block"]
  - 玩家HP: state["player"]["hp"]
  - 敌人: state["battle"]["enemies"]  (列表, 每项含 entity_id/name/hp/max_hp/block/intents/status)
  - 敌人意图: enemies[].intents (列表! 每项含 type/label/title/description)
"""

import re
import logging

logger = logging.getLogger("sts2_analyzer")

# 关键词识别：这些关键词出现在卡牌描述中表示该牌有 debuff 效果
DEBUFF_KEYWORDS = [
    "vulnerable", "weak", "frail", "poison",
    "易伤", "虚弱", "脆弱", "中毒",
    "strength down", "dexterity down",
]

BUFF_KEYWORDS = [
    "strength", "dexterity", "draw",
    "力量", "敏捷", "抽牌",
]


def _parse_damage_from_label(label: str) -> tuple:
    """从意图 label 中解析伤害数值
    
    STS2MCP 的 intent label 格式举例:
      - "12"        → 单次 12 伤害
      - "6x3"       → 3 次各 6 伤害 = 18 总伤害
      - "8x2"       → 2 次各 8 伤害 = 16 总伤害
    
    Returns:
        (total_damage, hits, per_hit_damage)
    """
    if not label:
        return (0, 0, 0)
    
    # 尝试匹配 "NxM" 格式 (多段攻击)
    multi_match = re.match(r"(\d+)\s*[xX×]\s*(\d+)", label)
    if multi_match:
        per_hit = int(multi_match.group(1))
        hits = int(multi_match.group(2))
        return (per_hit * hits, hits, per_hit)
    
    # 尝试匹配纯数字
    single_match = re.match(r"(\d+)", label)
    if single_match:
        dmg = int(single_match.group(1))
        return (dmg, 1, dmg)
    
    return (0, 0, 0)


def _has_debuff(description: str) -> bool:
    """检查卡牌描述中是否含有 debuff 关键词"""
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in DEBUFF_KEYWORDS)


def _has_buff(description: str) -> bool:
    """检查卡牌描述中是否含有 buff 关键词"""
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in BUFF_KEYWORDS)


def analyze_threats(game_state: dict) -> dict:
    """分析本回合所有敌人的威胁
    
    Returns:
        {
            "total_incoming_damage": int,
            "threats": [
                {
                    "entity_id": str,
                    "name": str,
                    "hp": int,
                    "max_hp": int,
                    "block": int,
                    "damage": int,
                    "hits": int,
                    "per_hit": int,
                    "intent_types": [str],
                    "is_attacking": bool,
                    "is_buffing": bool
                }
            ]
        }
    """
    battle = game_state.get("battle", {})
    enemies = battle.get("enemies", [])
    
    total_incoming = 0
    threats = []
    
    for enemy in enemies:
        intents = enemy.get("intents", [])
        intent_types = [i.get("type", "Unknown") for i in intents]
        
        enemy_damage = 0
        enemy_hits = 0
        enemy_per_hit = 0
        is_attacking = False
        is_buffing = False
        
        for intent in intents:
            intent_type = intent.get("type", "").lower()
            if "attack" in intent_type:
                is_attacking = True
                label = intent.get("label", "")
                dmg, hits, per_hit = _parse_damage_from_label(label)
                enemy_damage += dmg
                enemy_hits += hits
                enemy_per_hit = per_hit
            if "buff" in intent_type or "strategicbuff" in intent_type:
                is_buffing = True
        
        total_incoming += enemy_damage
        threats.append({
            "entity_id": enemy.get("entity_id"),
            "name": enemy.get("name"),
            "hp": enemy.get("hp", 0),
            "max_hp": enemy.get("max_hp", 0),
            "block": enemy.get("block", 0),
            "damage": enemy_damage,
            "hits": enemy_hits,
            "per_hit": enemy_per_hit,
            "intent_types": intent_types,
            "is_attacking": is_attacking,
            "is_buffing": is_buffing
        })
    
    return {
        "total_incoming_damage": total_incoming,
        "threats": threats
    }


def analyze_hand(game_state: dict) -> dict:
    """分析手牌的攻防能力（考虑费用限制）
    
    v2 新增: debuff_cards 类别（含有 Vulnerable/Weak 等的攻击/技能牌）
    
    Returns:
        {
            "energy": int,
            "max_energy": int,
            "total_playable_attack_damage": int,
            "total_playable_block": int,
            "attack_cards":  [{name, cost, damage_est, index, can_play, target_type, has_debuff}],
            "debuff_cards":  [{name, cost, damage_est, index, can_play, target_type}],
            "block_cards":   [{name, cost, block_est, index, can_play}],
            "power_cards":   [{name, cost, index, can_play}],
            "other_cards":   [{name, cost, index, can_play}]
        }
    """
    player = game_state.get("player", {})
    hand = player.get("hand", [])
    energy = player.get("energy", 0)
    max_energy = player.get("max_energy", 3)
    
    attack_cards = []
    debuff_cards = []
    block_cards = []
    power_cards = []
    other_cards = []
    
    for card in hand:
        card_type = card.get("type", "").lower()
        card_name = card.get("name", "Unknown")
        cost_str = card.get("cost", "0")
        can_play = card.get("can_play", False)
        index = card.get("index", 0)
        target_type = card.get("target_type", "")
        description = card.get("description", "")
        
        # 解析费用
        try:
            cost = int(cost_str) if cost_str != "X" else energy
        except (ValueError, TypeError):
            cost = 99
        
        # 从描述中提取伤害/格挡数值
        damage_est = 0
        block_est = 0
        
        dmg_match = re.search(r"[Dd]eal\s+(\d+)\s+damage|造成\s*(\d+)\s*点?\s*伤害", description)
        if dmg_match:
            damage_est = int(dmg_match.group(1) or dmg_match.group(2))
        
        blk_match = re.search(r"[Gg]ain\s+(\d+)\s+[Bb]lock|获得\s*(\d+)\s*点?\s*格挡", description)
        if blk_match:
            block_est = int(blk_match.group(1) or blk_match.group(2))
        
        has_debuff = _has_debuff(description)
        has_buff_effect = _has_buff(description)
        
        info = {
            "name": card_name,
            "cost": cost,
            "index": index,
            "can_play": can_play,
            "description": description
        }
        
        if card_type == "attack":
            info["damage_est"] = damage_est
            info["target_type"] = target_type
            info["has_debuff"] = has_debuff
            
            # 带 debuff 的攻击牌（如 Bash）同时放入两个列表
            if has_debuff:
                debuff_cards.append(info)
            attack_cards.append(info)
        elif card_type == "skill":
            if has_debuff:
                info["target_type"] = target_type
                info["damage_est"] = damage_est
                debuff_cards.append(info)
            elif block_est > 0:
                info["block_est"] = block_est
                block_cards.append(info)
            else:
                other_cards.append(info)
        elif card_type == "power":
            power_cards.append(info)
        else:
            other_cards.append(info)
    
    # 计算在费用限制内的最大攻击伤害
    playable_attacks = [c for c in attack_cards if c["can_play"]]
    playable_attacks.sort(key=lambda c: c["damage_est"] / max(c["cost"], 1), reverse=True)
    remaining_energy = energy
    total_atk = 0
    for c in playable_attacks:
        if c["cost"] <= remaining_energy:
            total_atk += c["damage_est"]
            remaining_energy -= c["cost"]
    
    # 计算在费用限制内的最大格挡
    playable_blocks = [c for c in block_cards if c["can_play"]]
    playable_blocks.sort(key=lambda c: c["block_est"] / max(c["cost"], 1), reverse=True)
    remaining_energy = energy
    total_blk = 0
    for c in playable_blocks:
        if c["cost"] <= remaining_energy:
            total_blk += c["block_est"]
            remaining_energy -= c["cost"]
    
    return {
        "energy": energy,
        "max_energy": max_energy,
        "total_playable_attack_damage": total_atk,
        "total_playable_block": total_blk,
        "attack_cards": attack_cards,
        "debuff_cards": debuff_cards,
        "block_cards": block_cards,
        "power_cards": power_cards,
        "other_cards": other_cards
    }


def analyze_potions(game_state: dict) -> dict:
    """分析当前持有的药水
    
    Returns:
        {
            "potions": [{slot, name, id, description, can_use_in_combat, target_type}],
            "has_healing_potion": bool,
            "has_block_potion": bool,
            "has_damage_potion": bool
        }
    """
    player = game_state.get("player", {})
    potions = player.get("potions", [])
    
    result = {
        "potions": potions,
        "has_healing_potion": False,
        "has_block_potion": False,
        "has_damage_potion": False
    }
    
    for p in potions:
        desc = p.get("description", "").lower()
        name = p.get("name", "").lower()
        pid = p.get("id", "").lower()
        combined = f"{desc} {name} {pid}"
        
        if any(kw in combined for kw in ["heal", "hp", "回复", "恢复", "治疗", "regen", "blood", "fairy"]):
            result["has_healing_potion"] = True
        if any(kw in combined for kw in ["block", "格挡", "shield", "protect", "ghost"]):
            result["has_block_potion"] = True
        if any(kw in combined for kw in ["damage", "伤害", "fire", "explosive", "attack", "poison"]):
            result["has_damage_potion"] = True
    
    return result


def generate_combat_briefing(game_state: dict) -> str:
    """生成拟人化的战场简报"""
    player = game_state.get("player", {})
    battle = game_state.get("battle", {})
    run = game_state.get("run", {})
    
    threat_info = analyze_threats(game_state)
    hand_info = analyze_hand(game_state)
    potion_info = analyze_potions(game_state)
    
    lines = []
    lines.append("═══════ 战场态势简报 ═══════")
    lines.append(f"第 {run.get('act', '?')} 幕 · 第 {run.get('floor', '?')} 层 · 回合 {battle.get('round', '?')}")
    lines.append("")
    
    # 我方状态
    lines.append(f"【我方】 {player.get('character', '?')}  "
                 f"HP: {player.get('hp', '?')}/{player.get('max_hp', '?')}  "
                 f"格挡: {player.get('block', 0)}  "
                 f"费用: {hand_info['energy']}/{hand_info['max_energy']}")
    
    # 药水
    if potion_info["potions"]:
        potion_names = [p.get("name", "?") for p in potion_info["potions"]]
        lines.append(f"  药水: {', '.join(potion_names)}")
    
    # 敌方简报
    lines.append("")
    lines.append("【敌方】")
    for t in threat_info["threats"]:
        intent_str = "/".join(t["intent_types"])
        dmg_str = ""
        if t["is_attacking"]:
            if t["hits"] > 1:
                dmg_str = f" → {t['per_hit']}x{t['hits']}={t['damage']}伤"
            else:
                dmg_str = f" → {t['damage']}伤"
        blk_str = f" 盾:{t['block']}" if t["block"] > 0 else ""
        buff_str = " 🔺正在BUFF" if t.get("is_buffing") else ""
        lines.append(f"  · {t['name']} [{t['entity_id']}]  HP:{t['hp']}/{t['max_hp']}{blk_str}  意图:{intent_str}{dmg_str}{buff_str}")
    
    # 威胁评估
    lines.append("")
    total_incoming = threat_info["total_incoming_damage"]
    current_block = player.get("block", 0)
    current_hp = player.get("hp", 0)
    net_damage = max(0, total_incoming - current_block)
    
    if total_incoming == 0:
        lines.append("📗 威胁等级: 安全 — 敌方本回合无攻击意图，全力输出！")
    elif net_damage >= current_hp:
        lines.append(f"🔴 威胁等级: 致命！ — 预期伤害 {total_incoming}，扣除现有格挡 {current_block}，"
                     f"净伤害 {net_damage} ≥ 当前HP {current_hp}！必须防御或使用药水！")
    elif net_damage > current_hp * 0.3:
        lines.append(f"🟠 威胁等级: 高危 — 预期伤害 {total_incoming}，建议积极防御。")
    else:
        lines.append(f"🟡 威胁等级: 可控 — 预期伤害 {total_incoming}，可攻可守。")
    
    # 手牌能力摘要
    lines.append("")
    lines.append(f"【手牌能力】 最大输出≈{hand_info['total_playable_attack_damage']}伤  "
                 f"最大格挡≈{hand_info['total_playable_block']}挡")
    if hand_info["debuff_cards"]:
        lines.append(f"  💀 Debuff牌: {', '.join(c['name'] for c in hand_info['debuff_cards'])}")
    if hand_info["power_cards"]:
        lines.append(f"  ⚡ 能力牌: {', '.join(c['name'] for c in hand_info['power_cards'])}")
    
    # 斩杀提醒
    for t in threat_info["threats"]:
        effective_hp = t["hp"] + t["block"]
        if effective_hp <= hand_info["total_playable_attack_damage"]:
            lines.append(f"⚔️ 斩杀机会: {t['name']} 有效血量 {effective_hp}（HP:{t['hp']}+盾:{t['block']}），"
                         f"可以击杀！")
    
    lines.append("═══════════════════════════")
    return "\n".join(lines)


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    mock_state = {
        "state_type": "monster",
        "run": {"act": 1, "floor": 3, "ascension": 0},
        "battle": {
            "round": 1, "turn": "player", "is_play_phase": True,
            "enemies": [
                {
                    "entity_id": "jaw_worm_0", "name": "Jaw Worm",
                    "hp": 42, "max_hp": 42, "block": 0, "status": [],
                    "intents": [{"type": "Attack", "label": "12", "title": "Chomp", "description": "Deals 12 damage."}]
                }
            ]
        },
        "player": {
            "character": "Ironclad", "hp": 75, "max_hp": 80, "block": 0,
            "energy": 3, "max_energy": 3,
            "hand": [
                {"index": 0, "name": "Strike", "type": "Attack", "cost": "1", "can_play": True,
                 "target_type": "AnyEnemy", "description": "Deal 6 damage."},
                {"index": 1, "name": "Strike", "type": "Attack", "cost": "1", "can_play": True,
                 "target_type": "AnyEnemy", "description": "Deal 6 damage."},
                {"index": 2, "name": "Defend", "type": "Skill", "cost": "1", "can_play": True,
                 "target_type": "Self", "description": "Gain 5 Block."},
                {"index": 3, "name": "Defend", "type": "Skill", "cost": "1", "can_play": True,
                 "target_type": "Self", "description": "Gain 5 Block."},
                {"index": 4, "name": "Bash", "type": "Attack", "cost": "2", "can_play": True,
                 "target_type": "AnyEnemy", "description": "Deal 8 damage. Apply 2 Vulnerable."}
            ],
            "gold": 99, "relics": [],
            "potions": [{"slot": 0, "name": "Blood Potion", "id": "blood_potion",
                         "description": "Heal 20% of your Max HP.", "can_use_in_combat": True,
                         "target_type": "Self"}]
        }
    }
    
    print(generate_combat_briefing(mock_state))
