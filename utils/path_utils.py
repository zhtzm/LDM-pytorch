import os


def generate_unique_filepath(base_path, class_name, state='train'):
    number = 0
    while True:
        run_path = f"{class_name}_{state}_{number}"
        run_path = os.path.join(base_path, run_path)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            return run_path
        number += 1


def set_default_workpath(func, script_path):
    try:
        work_path = func(script_path)
    except Exception as e:
        print(f"执行函数时发生错误: {e}")
    try:
        os.chdir(work_path)
    except FileNotFoundError:
        print(f"指定的目录 {work_path} 不存在。")

