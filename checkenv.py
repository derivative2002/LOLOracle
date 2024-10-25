import paddle

x = paddle.randn([10, 30])
linear = paddle.nn.Linear(30, 2)
y = linear(x)
print(y)

