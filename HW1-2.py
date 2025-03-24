from flask import Flask, render_template, request, jsonify, session
import numpy as np
import random

app = Flask(__name__)
app.secret_key = "secret"

# -----------------------------
# Iterative Policy Evaluation 參數
# -----------------------------
GAMMA = 0.9        # 折扣因子
MAX_ITER = 50      # 最大迭代次數
THRESHOLD = 1e-4   # 收斂閾值

def generate_random_policy(n, obstacles):
    """
    在每個「非障礙物」格子裡，隨機選一個方向 (↑, ↓, ←, →)。
    已被使用者選為障礙物的格子 => 設為 '■'。
    其中 obstacles 會是一個 set of (r, c) tuples。
    """
    directions = ["↑", "↓", "←", "→"]
    policy_arr = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            if (i, j) in obstacles:
                policy_arr[i][j] = '■'
            else:
                policy_arr[i][j] = random.choice(directions)
    return policy_arr

def policy_evaluation(n, policy_arr, obstacles):
    """
    針對給定 policy 做「迭代式策略評估 (Iterative Policy Evaluation)」：
      - 撞到障礙物或出界 => 留在原地 & 即時獎勵 = -1
      - 否則移動成功 => 即時獎勵 = 0
      - 障礙物格子 => V = None，不參與更新
      - 收斂條件 => delta < THRESHOLD 或超過 MAX_ITER
      - 使用 GAMMA = 0.9
    """
    # 先把所有格子初始化為 0
    value_arr = np.zeros((n, n), dtype=object)

    # 障礙物格子 => None
    for (r, c) in obstacles:
        value_arr[r][c] = None

    def in_bounds(i, j):
        return 0 <= i < n and 0 <= j < n

    def step(i, j):
        """
        從 (i, j) 依照 policy_arr[i][j] 移動一步。
        回傳 (next_state, reward)
        """
        # 若自己是障礙物，就不動
        if (i, j) in obstacles:
            return (i, j), 0

        action = policy_arr[i][j]
        if action == '■':
            # 與障礙物同樣視為無法移動
            return (i, j), 0

        # 根據方向決定下一步
        if action == '↑':
            new_i, new_j = i - 1, j
        elif action == '↓':
            new_i, new_j = i + 1, j
        elif action == '←':
            new_i, new_j = i, j - 1
        elif action == '→':
            new_i, new_j = i, j + 1
        else:
            return (i, j), 0

        # 撞牆 or 撞障礙物 => 留在原地 & reward = -1
        if not in_bounds(new_i, new_j) or (new_i, new_j) in obstacles:
            return (i, j), -1
        else:
            # 成功移動 => reward = 0
            return (new_i, new_j), 0

    # 迭代更新
    for _ in range(MAX_ITER):
        delta = 0
        new_value = value_arr.copy()

        for i in range(n):
            for j in range(n):
                # 障礙物 => 不更新
                if (i, j) in obstacles:
                    continue

                (ni, nj), reward = step(i, j)

                # 若下一個狀態是障礙物 => value=0
                if (ni, nj) in obstacles or value_arr[ni][nj] is None:
                    v_sprime = 0
                else:
                    v_sprime = value_arr[ni][nj]

                old_val = value_arr[i][j] if value_arr[i][j] is not None else 0
                updated_val = reward + GAMMA * v_sprime

                diff = abs(updated_val - old_val)
                if diff > delta:
                    delta = diff

                new_value[i][j] = round(updated_val, 2)

        value_arr = new_value
        # 收斂檢查
        if delta < THRESHOLD:
            break

    return value_arr

@app.route("/", methods=["GET", "POST"])
def index():
    """
    首頁：使用者輸入 n
    """
    if request.method == "POST":
        n = int(request.form.get("n"))
        session["n"] = n
        # 預設清空障礙物
        session["obstacles"] = []
        return render_template("select_obstacles.html", n=n)
    return render_template("index.html")

@app.route("/submit_obstacles", methods=["POST"])
def submit_obstacles():
    """
    接收使用者點選的障礙物清單 (list of lists)，轉成 list of tuples 存入 session
    以避免 "unhashable type: 'list'" 錯誤。
    """
    obstacles_json = request.form.get("obstacles")
    if obstacles_json:
        # obstacles_list 形如 [[0,0], [1,2], ...]
        obstacles_list = eval(obstacles_json)

        # 把每個 [r, c] 轉成 (r, c)
        obstacles_tuple_list = [tuple(item) for item in obstacles_list]

        # 存進 session
        session["obstacles"] = obstacles_tuple_list
    return jsonify({"status": "success"})

@app.route("/view_matrices")
def view_matrices():
    """
    1) 根據 session["n"] & session["obstacles"] 產生隨機 policy
    2) 對該 policy 做 iterative policy evaluation
    3) 障礙物格子在 Value= None, Policy='■' => 在模板顯示灰色
    """
    n = session.get("n", 5)
    # 這裡拿到的是 list of tuples
    obstacles_list = session.get("obstacles", [])

    # 轉成 set 方便查詢
    obstacles_set = set(obstacles_list)

    # 產生 policy
    policy_arr = generate_random_policy(n, obstacles_set)
    # 做策略評估
    value_arr = policy_evaluation(n, policy_arr, obstacles_set)

    # 轉成 Python list
    value_matrix = []
    policy_matrix = []
    for i in range(n):
        v_row = []
        p_row = []
        for j in range(n):
            v_row.append(value_arr[i][j])
            p_row.append(policy_arr[i][j])
        value_matrix.append(v_row)
        policy_matrix.append(p_row)

    return render_template(
        "view_matrices.html",
        n=n,
        value_matrix=value_matrix,
        policy_matrix=policy_matrix
    )

if __name__ == "__main__":
    app.run(debug=True)
