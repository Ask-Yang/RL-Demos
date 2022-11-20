import numpy as np

#以12ms作为分段标准
ms_per_part = 12 


#计算一个动作需要分为几段
def get_motion_part_num( time ):
    part = int (time  / ms_per_part)
    if time  % ms_per_part != 0:
        part += 1 
    return part 

#根据段数将当前动作进行分解
#输入为当前角度列表 目标角度列表  分段数量
#输出为分段角度列表
def split_motion (current_joint, motion ,  part):
    step = (motion - current_joint )/part
    #print(step)
    t = 0 
    for i in step:
        step[t] = round(i, 4)
        t += 1 
    #print(step)
    return step

motion = np.array([1,1,1,1])

current_joint = np.array([2,3,4,5])
#print(current_joint - motion)
#print((current_joint - motion)/10)
split_motion(current_joint,motion,12)
