import paddle

def check_environment():
    print("PaddlePaddle 版本:", paddle.__version__)
    print("是否支持 GPU:", paddle.is_compiled_with_cuda())

    if paddle.is_compiled_with_cuda():
        gpu_device = paddle.CUDAPlace(0)
        paddle.set_device('gpu:0')
        x = paddle.randn([10, 10])
        y = paddle.randn([10, 10])
        result = paddle.mm(x, y)
        print("GPU 计算测试通过")
    else:
        cpu_device = paddle.CPUPlace()
        paddle.set_device('cpu')
        x = paddle.randn([10, 10])
        y = paddle.randn([10, 10])
        result = paddle.mm(x, y)
        print("CPU 计算测试通过")

if __name__ == '__main__':
    check_environment()
