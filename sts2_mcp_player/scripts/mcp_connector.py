"""
STS2MCP 通讯连接器 - 基于 STS2MCP 真实 API 规范重写
GitHub: https://github.com/Gennadiyev/STS2MCP

端口: 15526 (Mod 默认)
基础路径: /api/v1/singleplayer
获取状态: GET + ?format=json
执行动作: POST + JSON body
"""

import json
import logging
import sys
import os
import time

# 优先尝试加载 requests，如果没有则回退到 urllib
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False

logger = logging.getLogger("sts2_connector")

# ============================================================
# 配置
# ============================================================
BASE_URL = "http://localhost:15526"
SP_ENDPOINT = f"{BASE_URL}/api/v1/singleplayer"
TIMEOUT = 10  # 秒


# ============================================================
# 底层 HTTP 封装
# ============================================================
def _get(params: dict = None) -> dict:
    """发送 GET 请求获取游戏状态"""
    url = SP_ENDPOINT
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"

    if HAS_REQUESTS:
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError:
            return {"error": "无法连接到 STS2MCP Mod。请确认游戏已启动且 Mod 已加载。"}
        except Exception as e:
            return {"error": str(e)}
    else:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e)}


def _post(body: dict) -> dict:
    """发送 POST 请求执行游戏动作"""
    if HAS_REQUESTS:
        try:
            r = requests.post(SP_ENDPOINT, json=body, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError:
            return {"error": "无法连接到 STS2MCP Mod。请确认游戏已启动且 Mod 已加载。"}
        except Exception as e:
            return {"error": str(e)}
    else:
        try:
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                SP_ENDPOINT, data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e)}


# ============================================================
# 游戏状态查询
# ============================================================
def get_game_state(fmt: str = "json") -> dict:
    """获取完整游戏状态（战斗/地图/商店/事件等全界面）
    
    Args:
        fmt: "json" 返回结构化数据, "markdown" 返回人类可读文本
    Returns:
        包含 state_type, player, battle 等字段的完整游戏状态字典
    """
    logger.info("正在获取游戏状态...")
    return _get({"format": fmt})


def check_connection() -> bool:
    """检测 STS2MCP Mod 是否在线"""
    result = _get()
    if "error" in result:
        return False
    # 根目录返回 {"message": "Hello from STS2 MCP v...", "status": "ok"}
    return result.get("status") == "ok" or "state_type" in result


# ============================================================
# 战斗动作
# ============================================================
def combat_play_card(card_index: int, target: str = None) -> dict:
    """打出一张手牌
    
    Args:
        card_index: 手牌中的 0-based 索引（注意：出牌后索引会变化！建议从右往左出牌）
        target: 目标敌人的 entity_id，如 "jaw_worm_0"。单体攻击牌必须指定。
    """
    body = {"action": "play_card", "card_index": card_index}
    if target is not None:
        body["target"] = target
    logger.info(f"出牌: 索引={card_index}, 目标={target or '无'}")
    return _post(body)


def combat_end_turn() -> dict:
    """结束当前回合"""
    logger.info("结束回合")
    return _post({"action": "end_turn"})


def combat_select_card(card_index: int) -> dict:
    """战斗中选牌（如"选择一张牌丢弃/消耗"的提示）"""
    return _post({"action": "combat_select_card", "card_index": card_index})


def combat_confirm_selection() -> dict:
    """确认战斗中的选牌"""
    return _post({"action": "combat_confirm_selection"})


# ============================================================
# 药水
# ============================================================
def use_potion(slot: int, target: str = None) -> dict:
    """使用药水
    
    Args:
        slot: 药水槽位索引
        target: 目标敌人 entity_id（仅针对敌方的药水需要）
    """
    body = {"action": "use_potion", "slot": slot}
    if target is not None:
        body["target"] = target
    logger.info(f"使用药水: 槽位={slot}, 目标={target or '无'}")
    return _post(body)


def discard_potion(slot: int) -> dict:
    """丢弃药水腾出空间"""
    return _post({"action": "discard_potion", "slot": slot})


# ============================================================
# 地图
# ============================================================
def map_choose_node(node_index: int) -> dict:
    """选择地图节点（从 next_options 列表中的 0-based 索引）"""
    logger.info(f"选择地图节点: 索引={node_index}")
    return _post({"action": "choose_map_node", "index": node_index})


# ============================================================
# 篝火
# ============================================================
def rest_choose_option(option_index: int) -> dict:
    """篝火选择（休息/升级/挖掘等）"""
    logger.info(f"篝火选择: 索引={option_index}")
    return _post({"action": "choose_rest_option", "index": option_index})


# ============================================================
# 奖励
# ============================================================
def rewards_claim(reward_index: int) -> dict:
    """领取战斗奖励（金币/药水/遗物直接领取，卡牌会打开选卡界面）"""
    return _post({"action": "claim_reward", "index": reward_index})


def rewards_pick_card(card_index: int) -> dict:
    """从卡牌奖励中选择一张加入牌组"""
    return _post({"action": "select_card_reward", "card_index": card_index})


def rewards_skip_card() -> dict:
    """跳过卡牌奖励"""
    return _post({"action": "skip_card_reward"})


def proceed_to_map() -> dict:
    """从当前界面（奖励/篝火/商店）返回地图"""
    return _post({"action": "proceed"})


# ============================================================
# 商店
# ============================================================
def shop_purchase(item_index: int) -> dict:
    """购买商店物品"""
    return _post({"action": "shop_purchase", "index": item_index})


# ============================================================
# 事件
# ============================================================
def event_choose_option(option_index: int) -> dict:
    """选择事件选项"""
    return _post({"action": "choose_event_option", "index": option_index})


def event_advance_dialogue() -> dict:
    """推进远古事件对话"""
    return _post({"action": "advance_dialogue"})


# ============================================================
# 卡牌选择（升级/变换/移除等界面）
# ============================================================
def deck_select_card(card_index: int) -> dict:
    """在选卡界面中选中/取消选中一张牌"""
    return _post({"action": "select_card", "index": card_index})


def deck_confirm_selection() -> dict:
    """确认选卡"""
    return _post({"action": "confirm_selection"})


def deck_cancel_selection() -> dict:
    """取消选卡"""
    return _post({"action": "cancel_selection"})


# ============================================================
# 遗物选择
# ============================================================
def relic_select(relic_index: int) -> dict:
    """选择遗物（如 BOSS 遗物奖励）"""
    return _post({"action": "select_relic", "index": relic_index})


def relic_skip() -> dict:
    """跳过遗物选择"""
    return _post({"action": "skip_relic_selection"})


# ============================================================
# 宝箱
# ============================================================
def treasure_claim_relic(relic_index: int) -> dict:
    """领取宝箱中的遗物"""
    return _post({"action": "claim_treasure_relic", "index": relic_index})


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    
    print("=" * 50)
    print("STS2MCP 连接测试")
    print("=" * 50)
    
    # 先测试根路径
    print("\n1. 测试 Mod 根路径连接...")
    if HAS_REQUESTS:
        try:
            r = requests.get(BASE_URL, timeout=5)
            print(f"   ✅ Mod 在线: {r.json()}")
        except Exception as e:
            print(f"   ❌ 连接失败: {e}")
            print("   请确保: (1) 杀戮尖塔2 已启动 (2) STS2_MCP Mod 已启用")
            sys.exit(1)
    
    # 获取游戏状态
    print("\n2. 获取游戏状态...")
    state = get_game_state("json")
    if "error" in state:
        print(f"   ❌ {state['error']}")
    else:
        state_type = state.get("state_type", "unknown")
        print(f"   ✅ 当前界面: {state_type}")
        if "player" in state:
            p = state["player"]
            print(f"   角色: {p.get('character')} | HP: {p.get('hp')}/{p.get('max_hp')} | 金币: {p.get('gold')}")
        if "battle" in state:
            b = state["battle"]
            print(f"   回合数: {b.get('round')} | 轮次: {b.get('turn')}")
            for e in b.get("enemies", []):
                print(f"   敌人: {e.get('name')} HP={e.get('hp')}/{e.get('max_hp')} 意图={[i.get('type') for i in e.get('intents', [])]}")
        print(f"\n   完整状态 JSON 前 500 字符:\n   {json.dumps(state, ensure_ascii=False)[:500]}")
