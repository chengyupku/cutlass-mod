def create_sending_info(block):
    # 获取维度大小
    x_size = len(block)
    y_size = len(block[0])
    k_rounds = len(block[0][0])

    # 初始化发送信息列表
    sending_info = [[[[] for _ in range(k_rounds)] for _ in range(y_size)] for _ in range(x_size)]

    # 遍历每个block[x][y]
    for x in range(x_size):
        for y in range(y_size):
            # 对于每个block[x][y]中的k轮
            for k in range(k_rounds):
                # 获取目标block坐标
                target_x, target_y = block[x][y][k]
                # 将当前block的坐标添加到目标block的发送列表中
                sending_info[target_x][target_y][k].append((x, y))

    return sending_info

# 示例数据
block = [
    [
        [[1, 1], [1, 1]], # block[0][0] 两轮发送信息
        [[1, 0], [1, 1]]
    ],
    [
        [[0, 1], [0, 0]],
        [[0, 0], [0, 1]]
    ]
]

# 获取发送信息
sending_info = create_sending_info(block)

# 输出结果以验证
for i, row in enumerate(sending_info):
    for j, col in enumerate(row):
        print(f"Block [{i}][{j}] sends to: {col}")