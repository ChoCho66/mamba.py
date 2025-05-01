import builtins
import functools

# 儲存原始的 print 函數
_original_print = builtins.print

# 全局 debug 變量
debug = True

# 定義新的 print 函數
@functools.wraps(_original_print)
def custom_print(*args, **kwargs):
    if debug:
        return _original_print(*args, **kwargs)
    return None

# 替換原始的 print 函數
builtins.print = custom_print

# 測試範例
if __name__ == "__main__":
    print("測試 debug = True")  # 會打印
    debug = False
    print("測試 debug = False")  # 不會打印
    debug = True
    print("測試 debug = True 再次")  # 會打印