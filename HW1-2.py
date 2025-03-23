from flask import Flask, render_template, request
import numpy as np
import random

app = Flask(__name__)

# -----------------------------
# 1) 定義一些參數，可自行調整
# -----------------------------
GAMMA = 0.9          # 折扣因子
MAX_ITER = 50        # 最大迭代次數
THRESHOLD = 1e-4     # 收斂閾值

# 假設「嘗試移動到障礙物或網格外」就留在原地並 -1 獎勵，其餘動作成功且 0 獎勵。
# 你可視需求改為每走一步 -0.1 等等。

def generate_random_policy(n):
    """
    在每個可走的格子中，隨機選一個方向 (↑, ↓, ←, →)。
    障礙物格子用 '■' 表示，沒有動作。
    傳回 policy_arr (np array of objects) 和 obstacle_positions (set)。
    """
    directions = ["↑", "↓", "←", "→"]
    policy_arr = np.random.choice(directions, size=(n, n)).astype(object)

    # 隨機選 (n-2) 個障礙物
    obstacle_positions = set()
    while len(obstacle_positions) < n - 2:
        obstacle_positions.add((random.randint(0, n - 1), random.randint(0, n - 1)))

    # 障礙物的格子，Policy 設為 '■'
    for (i, j) in obstacle_positions:
        policy_arr[i][j] = '■'

    return policy_arr, obstacle_positions


def policy_evaluation(n, policy_arr, obstacle_positions):
    """
    使用「迭代策略評估 (Iterative Policy Evaluation)」計算每個格子的 Value。
    簡化假設：
      - 確定性移動：Policy 指示往哪走，就往那走（若沒有障礙或超界）。
      - 若移動方向是障礙物或牆壁 => 留在原地並得到 -1。
      - 否則 => 移到新格子並得到 0。
      - 折扣因子 gamma = GAMMA
      - 初始 V(s)=0，迭代直到收斂或到達最大回合。
    """
    # 初始化 V 為全 0；型別設 object 才能與 None 做區分
    value_arr = np.zeros((n, n), dtype=object)

    # 將障礙物位置的 Value 設為 None (不參與計算，也顯示灰色)
    for (i, j) in obstacle_positions:
        value_arr[i][j] = None

    def in_bounds(i, j):
        return 0 <= i < n and 0 <= j < n

    # 幫助我們取得 (下一個狀態) 與 (獎勵)
    # 根據 policy_arr[i][j] 的動作，決定要移動到哪裡
    def step(i, j):
        # 如果本身是障礙物，就沒有任何動作 => 留在原地
        if (i, j) in obstacle_positions:
            return (i, j), 0  # Value 也會是 None，稍後不參與迭代

        action = policy_arr[i][j]
        if action == '■':
            # 同樣代表障礙物，留在原地
            return (i, j), 0

        # 決定目標位置
        if action == '↑':
            new_i, new_j = i - 1, j
        elif action == '↓':
            new_i, new_j = i + 1, j
        elif action == '←':
            new_i, new_j = i, j - 1
        elif action == '→':
            new_i, new_j = i, j + 1
        else:
            # 不預期的情況，留在原地
            return (i, j), 0

        # 檢查是否出界 或 走到障礙物
        if not in_bounds(new_i, new_j) or (new_i, new_j) in obstacle_positions:
            # 撞到障礙物或出界 => 留在原地並 -1 懲罰
            return (i, j), -1
        else:
            # 正常走成功 => 0 獎勵
            return (new_i, new_j), 0

    # 迭代策略評估
    for _ in range(MAX_ITER):
        delta = 0
        new_value = value_arr.copy()  # 先複製，準備更新

        for i in range(n):
            for j in range(n):
                # 障礙物位置 => 不更新
                if (i, j) in obstacle_positions:
                    continue

                # 取得下一個狀態與獎勵
                (ni, nj), reward = step(i, j)

                # 若下一個狀態是障礙物 => 其 value_arr[ni][nj] = None，不參與計算
                if (ni, nj) in obstacle_positions:
                    # 只考慮該 step 的 即時獎勵
                    v_sprime = 0
                else:
                    # 否則 => 取舊 V(s')
                    v_sprime = value_arr[ni][nj]
                    if v_sprime is None:
                        # 理論上不會 None，除非 next state 也是障礙物
                        v_sprime = 0

                # Bellman update for policy evaluation
                # V(s) = R(s,a,s') + gamma * V(s')
                updated_val = reward + GAMMA * v_sprime

                # 計算變動量
                old_val = value_arr[i][j]
                if old_val is None:
                    old_val = 0
                diff = abs(updated_val - old_val)
                if diff > delta:
                    delta = diff

                new_value[i][j] = round(updated_val, 2)

        value_arr = new_value
        if delta < THRESHOLD:
            # 收斂
            break

    # 回傳最終收斂後的 Value 陣列
    return value_arr


def generate_value_policy_matrices(n):
    """
    1) 隨機產生 Policy
    2) 用策略評估算出 Value
    3) 將結果轉成 Python list 返回
    """
    # (1) 隨機產生 policy
    policy_arr, obstacle_positions = generate_random_policy(n)

    # (2) 策略評估 => 得到 V(s)
    value_arr = policy_evaluation(n, policy_arr, obstacle_positions)

    # 轉成 Python list
    value_matrix = []
    policy_matrix = []

    for i in range(n):
        v_row = []
        p_row = []
        for j in range(n):
            v = value_arr[i][j]
            p = policy_arr[i][j]
            v_row.append(v)
            p_row.append(p)
        value_matrix.append(v_row)
        policy_matrix.append(p_row)

    return value_matrix, policy_matrix


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        n = int(request.form.get("n"))
        value_matrix, policy_matrix = generate_value_policy_matrices(n)
        return render_template("view_matrices.html", n=n, value_matrix=value_matrix, policy_matrix=policy_matrix)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
