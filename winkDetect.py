import numpy as np
import cv2
import dlib
import math
import datetime
import matplotlib.pyplot as plt

#每分钟眨眼频率
highWinkRate = 30
lowWinkRate = 10

#判断是否睁眼和闭眼的阈值
highEyeWidth = 0.27
lowEywWidth = 0.24

#多少分钟判断一次疲劳度
minuteNum = 1
#判断了多少次疲劳度
minuteBunchNum = 1

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#画图的列表
ifWink0 = []
ifWink1 = []
countWinkList = []
def pointLen(p1, p2):
    len = math.sqrt(abs(p1[0]-p2[0])*abs(p1[0]-p2[0])+abs(p1[1]-p2[1])*abs(p1[1]-p2[1]))
    return len
# cv2读取图像
cap = cv2.VideoCapture(0)
lastWink = 0
countWink = 0
countWink1 = 0
countWink2 = 0
winkChazhi = 0

#时间变量
time0 = datetime.datetime.now()
time0_2 = datetime.datetime.now()
time1 = 0
time3 = 0
time3_2 = 0


while(cap.isOpened()):
    ret,img=cap.read()
    cv2.imshow("capture", img)
    k = cv2.waitKey(1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 0)
    totalpos = []
    #标注关键点
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            totalpos.append(pos)
    if(totalpos==[]):
        print("未检测到人脸！可能人脸被遮挡！等待2秒后摄像头会重新开始捕捉检测。")
        cv2.waitKey(2000)
        continue
    #两眼左右长度
    leftEyeLength = (pointLen(totalpos[39], totalpos[36]))
    rightEyeLength = (pointLen(totalpos[42], totalpos[45]))
    #两眼上下长度
    leftEyeWidth1 = (pointLen(totalpos[37], totalpos[41]))
    leftEyeWidth2 = (pointLen(totalpos[38], totalpos[40]))
    rightEyeWidth1 = (pointLen(totalpos[43], totalpos[47]))
    rightEyeWidth2 = (pointLen(totalpos[44], totalpos[46]))
    leftEyeWidth = (leftEyeWidth1 + leftEyeWidth2)/2
    rightEyeWidth = (rightEyeWidth1 + rightEyeWidth2)/2

    # print(leftEyeLength)
    # print(rightEyeLength)
    # print(leftEyeWidth)
    # print(rightEyeWidth)
    ifWinkLeft = leftEyeWidth/leftEyeLength
    ifWinkRight = rightEyeWidth/rightEyeLength

    ifWink0.append(ifWinkLeft)
    ifWink1.append(ifWinkRight)
    # print(leftEyeWidth/leftEyeLength)
    # print(rightEyeWidth/rightEyeLength)
    countWinkBefore = countWink
    #判断是否睁眼状态
    if (ifWinkLeft > highEyeWidth and ifWinkRight > highEyeWidth):
        lastWink = 0
    if(ifWinkLeft < lowEywWidth and ifWinkRight < lowEywWidth):
        if(lastWink != 1):
            if (time1 == 0):
                time1 = datetime.datetime.now()
            else:
                time2 = datetime.datetime.now()
                print("眨眼间隔为：" + str(abs(time1 - time2).total_seconds()) + "秒")
                time1 = time2
            print(leftEyeWidth/leftEyeLength)
            print(rightEyeWidth/rightEyeLength)
            print("眨眼了一次！")
            countWink = countWink + 1

        lastWink = 1

    cv2.waitKey(1)

    if(countWinkBefore !=countWink):
        print("目前眨眼次数为" + str(countWink) + "次！")
    time3 = datetime.datetime.now()
    time3_2 = datetime.datetime.now()

    #计时
    if(abs(time3_2 - time0_2).total_seconds() > 60):
        countWinkList.append(countWink-countWink1)
        time0_2 = time3_2
        countWink1 = countWink

    if(abs(time3 - time0).total_seconds() > 60*minuteNum):
        winkChazhi = countWink-countWink2
        # print((countWink))
        # print(countWink2)
        if(winkChazhi>highWinkRate*minuteNum):
            print("检测到上" + str(minuteNum) + " 分钟您的眨眼频率过高。您的眨眼次数为 "
                  + str(winkChazhi/minuteNum) + " 次每分钟。正常眨眼次数为每分钟之内 " + str(highWinkRate) +
                  " 次。这代表您可能需要休息一下眼睛~")
        elif (winkChazhi <lowWinkRate*minuteNum):
            print("检测到上" + str(minuteNum) + " 分钟您的眨眼频率过低。您的眨眼次数为 "
                  + str(winkChazhi / minuteNum) + " 次每分钟。正常眨眼次数为每分钟 " + str(lowWinkRate) +
                  " 次以上。您可能聚精会神地看太久啦~请眨眨眼湿润一下眼球叭。保护好眼睛哦~")
        else:
            print("检测到上" + str(minuteNum) + " 分钟您的眨眼频率属于正常范畴。您的眨眼次数为 "
                  + str(winkChazhi/minuteNum)+ " 次每分钟。")
        print("您的上"+ str(minuteBunchNum*minuteNum) +"分钟的平均眨眼频率为"+ str(countWink/minuteBunchNum)+"次每分钟。")
        time0 = datetime.datetime.now()
        minuteBunchNum = minuteBunchNum + 1
        countWink2 = countWink
    if(countWink>100):
        break
# print(11111111111111111)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.subplot(211)
X = np.arange(0,len(ifWink0))
Y = np.arange(1,len(countWinkList)+1)
# print(X.shape)
# print(ifWink0)
plt.plot(X, ifWink0)
plt.plot(X, ifWink1)
plt.xlabel('眨眼次数', fontsize=16)
plt.ylabel('眼部纵横比', fontsize=16)
plt.subplot(212)
plt.plot(Y, countWinkList)
plt.xlabel('分钟', fontsize=16)
plt.ylabel('眨眼次数', fontsize=16)
plt.savefig("fig1.jpg")
plt.show()
