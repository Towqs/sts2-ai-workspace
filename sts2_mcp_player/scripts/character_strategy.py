"""
角色策略配置 - 根据不同角色的被动技能调整打法

STS2 的角色及其核心机制:
  - Ironclad (战士):  被动回血(每场战斗结束+6HP), 力量流, 重型攻击
  - Silent (猎人):    被动(每场开局多抽2), 毒液/碎片流, 多段攻击
  - Defect (缺陷体):  被动(球位), 闪电球/冰球/暗黑球
  - Watcher (观察者): 被动(姿态), 忿怒/平静切换
  - Necrobinder:      被动(宠物 Osty), 死亡触发
  - The Regent:       被动(星星资源), 星星消耗牌
"""

import logging
logger = logging.getLogger("sts2_character")

# ============================================================
# 角色策略档案
# ============================================================
CHARACTER_PROFILES = {
    # === 战士 (Ironclad) ===
    "ironclad": {
        "name_cn": "战士",
        "passive": "每场战斗结束后回复6点HP",
        "strategy": "力量叠加 + 重型攻击。因为有被动回血，可以承受更多伤害换取输出。",
        # 选卡优先级 (描述关键词)
        "preferred_keywords": [
            "strength", "力量",           # 力量牌最优先
            "exhaust", "消耗",            # 消耗流
            "vulnerable", "易伤",         # 减防
            "heavy", "重击",
        ],
        "avoid_keywords": [],
        # 战斗风格
        "aggression": 0.7,     # 0~1, 越高越倾向进攻
        "hp_rest_threshold": 0.5,  # HP 低于此比例时篝火选休息
        "elite_hp_threshold": 0.65,  # HP 高于此比例时才挑战精英
    },
    
    # === 猎人 (Silent) ===
    "silent": {
        "name_cn": "猎人",
        "passive": "每场战斗开始额外抽2张牌",
        "strategy": "毒液叠加 + 碎片(Shiv)流。利用多抽被动打连击，毒是核心输出。",
        "preferred_keywords": [
            "poison", "毒", "中毒",       # 毒最优先
            "shiv", "碎片",               # 碎片流
            "weak", "虚弱",               # 减攻
            "draw", "抽",                 # 抽牌
            "discard", "弃",              # 弃牌触发
        ],
        "avoid_keywords": [],
        "aggression": 0.5,
        "hp_rest_threshold": 0.55,
        "elite_hp_threshold": 0.7,
    },
    
    # === 缺陷体 (Defect) ===
    "defect": {
        "name_cn": "缺陷体",
        "passive": "拥有球位(Orb)系统",
        "strategy": "闪电球输出 + 冰球防御。能力牌(Power)极其重要，优先打。球位管理是关键。",
        "preferred_keywords": [
            "orb", "球",
            "lightning", "闪电",
            "frost", "冰",
            "focus", "集中",
            "channel", "引导",
            "evoke", "唤出",
        ],
        "avoid_keywords": [],
        "aggression": 0.4,  # 偏防守，靠球输出
        "hp_rest_threshold": 0.55,
        "elite_hp_threshold": 0.7,
    },
    
    # === 观察者 (Watcher) ===
    "watcher": {
        "name_cn": "观察者",
        "passive": "姿态系统(忿怒/平静)",
        "strategy": "忿怒姿态造成双倍伤害但受到双倍伤害。核心是切忿怒→打一波→切回平静。",
        "preferred_keywords": [
            "wrath", "忿怒",
            "calm", "平静",
            "stance", "姿态",
            "mantra", "真言",
            "divinity", "神性",
            "retain", "保留",
            "scry", "预见",
        ],
        "avoid_keywords": [],
        "aggression": 0.6,
        "hp_rest_threshold": 0.5,
        "elite_hp_threshold": 0.65,
    },
    
    # === 死灵绑定者 (Necrobinder) ===
    "necrobinder": {
        "name_cn": "死灵绑定者",
        "passive": "拥有宠物 Osty",
        "strategy": "死亡触发 + 宠物协同。利用消耗和死亡效果获取价值。",
        "preferred_keywords": [
            "death", "死亡",
            "exhaust", "消耗",
            "soul", "灵魂",
            "pet", "宠物",
            "summon", "召唤",
        ],
        "avoid_keywords": [],
        "aggression": 0.5,
        "hp_rest_threshold": 0.55,
        "elite_hp_threshold": 0.65,
    },
    
    # === 摄政王 (The Regent) ===
    "regent": {
        "name_cn": "摄政王",
        "passive": "星星(Stars)资源系统",
        "strategy": "管理星星资源。部分强力牌消耗星星，需要合理分配。",
        "preferred_keywords": [
            "star", "星",
            "crown", "皇冠",
            "royal", "皇家",
            "decree", "法令",
        ],
        "avoid_keywords": [],
        "aggression": 0.5,
        "hp_rest_threshold": 0.55,
        "elite_hp_threshold": 0.65,
    },
}

# 默认策略（未识别角色时使用）
DEFAULT_PROFILE = {
    "name_cn": "未知角色",
    "passive": "未知",
    "strategy": "稳健打法：攻守兼备。",
    "preferred_keywords": [],
    "avoid_keywords": [],
    "aggression": 0.5,
    "hp_rest_threshold": 0.55,
    "elite_hp_threshold": 0.7,
}


def get_character_profile(character_name: str) -> dict:
    """根据角色名获取策略档案（支持中英文名）"""
    if not character_name:
        return DEFAULT_PROFILE
    
    name_lower = character_name.lower().strip()
    
    # 直接匹配英文 key
    if name_lower in CHARACTER_PROFILES:
        return CHARACTER_PROFILES[name_lower]
    
    # 中文名映射表（游戏内显示的中文名 → profile key）
    cn_name_map = {
        "铁甲战士": "ironclad",
        "战士": "ironclad",
        "猎人": "silent",
        "沉默猎手": "silent",
        "缺陷体": "defect",
        "机器人": "defect",
        "观察者": "watcher",
        "死灵绑定者": "necrobinder",
        "死灵师": "necrobinder",
        "储君": "regent",
        "摄政王": "regent",
    }
    
    for cn_name, key in cn_name_map.items():
        if cn_name in character_name:
            profile = CHARACTER_PROFILES[key]
            logger.info(f"角色匹配: '{character_name}' -> {profile['name_cn']} ({key})")
            return profile
    
    # 英文模糊匹配
    for key, profile in CHARACTER_PROFILES.items():
        if key in name_lower or name_lower in key:
            return profile
    
    logger.warning(f"未识别的角色: '{character_name}'，使用默认策略")
    return DEFAULT_PROFILE


def analyze_deck_needs(deck_cards: list) -> dict:
    """分析牌组构成，判断缺什么类型的牌
    
    Args:
        deck_cards: 牌组中所有牌的列表 (从 draw_pile + discard_pile + hand + exhaust)
    
    Returns:
        dict with counts and recommendations
    """
    attack_count = 0
    block_count = 0  
    power_count = 0
    skill_count = 0
    total = len(deck_cards) if deck_cards else 1
    
    for card in deck_cards:
        card_type = card.get("type", "").lower()
        card_name = card.get("name", "").lower()
        desc = card.get("description", "").lower()
        
        if card_type == "attack":
            attack_count += 1
        elif card_type == "power":
            power_count += 1
        elif card_type == "skill":
            skill_count += 1
            # 进一步分类: 格挡 vs 其他技能
            if any(kw in desc for kw in ["block", "格挡", "防御"]) or any(kw in card_name for kw in ["defend", "防御"]):
                block_count += 1
    
    attack_ratio = attack_count / total
    block_ratio = block_count / total
    power_ratio = power_count / total
    
    # 判断缺什么
    needs = {
        "total": total,
        "attack_count": attack_count,
        "block_count": block_count,
        "power_count": power_count,
        "skill_count": skill_count,
        "attack_ratio": attack_ratio,
        "block_ratio": block_ratio,
        "need_attack": attack_ratio < 0.35,    # 攻击牌不到35%，缺输出
        "need_block": block_ratio < 0.15,      # 格挡牌不到15%，缺防御  
        "need_power": power_count < 2,          # 能力牌不到2张，缺长期增益
        "need_aoe": total > 12 and attack_count < 3,  # 后期缺AOE
    }
    
    return needs


def score_card_for_character(card: dict, profile: dict, deck_needs: dict = None) -> float:
    """根据角色策略 + 牌组需求给卡牌打分
    
    Args:
        card: 待评估的卡牌
        profile: 角色策略档案
        deck_needs: 牌组需求分析 (from analyze_deck_needs)
    
    Returns:
        分数越高越好
    """
    score = 0.0
    description = card.get("description", "").lower()
    card_name = card.get("name", "").lower()
    card_type = card.get("type", "").lower()
    rarity = card.get("rarity", "").lower()
    combined = f"{description} {card_name}"
    
    # 稀有度加分
    if rarity == "rare":
        score += 3.0
    elif rarity == "uncommon":
        score += 1.5
    
    # 角色偏好关键词匹配
    for kw in profile.get("preferred_keywords", []):
        if kw.lower() in combined:
            score += 2.0
    
    # 避免关键词
    for kw in profile.get("avoid_keywords", []):
        if kw.lower() in combined:
            score -= 3.0
    
    # 能力牌(Power)通常都值得拿
    if card_type == "power":
        score += 1.5
    
    # 0费牌加分
    cost_str = card.get("cost", "1")
    try:
        if int(cost_str) == 0:
            score += 1.0
    except (ValueError, TypeError):
        pass
    
    # ===== 牌组需求加分 =====
    if deck_needs:
        # 缺攻击牌 → 攻击牌加分
        if deck_needs.get("need_attack") and card_type == "attack":
            score += 2.0
        
        # 缺防御牌 → 格挡技能加分
        if deck_needs.get("need_block"):
            if any(kw in combined for kw in ["block", "格挡", "防御", "defend"]):
                score += 2.0
        
        # 缺能力牌 → 能力牌额外加分
        if deck_needs.get("need_power") and card_type == "power":
            score += 1.5
        
        # 已经有很多攻击牌了 → 攻击牌减分
        if not deck_needs.get("need_attack") and deck_needs.get("attack_ratio", 0) > 0.5:
            if card_type == "attack":
                score -= 1.0
        
        # AOE/多目标加分（对多怪战有价值）
        if any(kw in combined for kw in ["all enem", "所有敌人", "all"]):
            score += 1.5
    
    # 通用好牌加分（抽牌、降费等）
    if any(kw in combined for kw in ["draw", "抽", "额外抽"]):
        score += 1.5
    if any(kw in combined for kw in ["upgrade", "升级"]):
        score += 1.0
    
    return score


def should_skip_card_reward(cards: list, profile: dict, deck_size: int = 10, deck_needs: dict = None) -> bool:
    """判断是否应该跳过卡牌奖励（防止牌组膨胀）"""
    # 牌组太大时提高跳过阈值
    if deck_size > 25:
        best_score = max(score_card_for_character(c, profile, deck_needs) for c in cards)
        return best_score < 2.5  # 25+张只拿高分牌
    elif deck_size > 20:
        best_score = max(score_card_for_character(c, profile, deck_needs) for c in cards)
        return best_score < 1.5
    
    return False  # 牌组小的时候尽量拿

