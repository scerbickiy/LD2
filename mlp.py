import numpy as np
import random
import matplotlib.pyplot as plt

x_vector = np.arange(0.1, 1.0 + 1/22, 1/22)
y_exp = (1 + 0.6 * np.sin (2 * np.pi * x_vector / 0.7) + 0.3 * np.sin (2 * np.pi * x_vector)) / 2
# print(y_exp)
# plt.plot(x_vector, y)
# plt.show()

#first layer weights
w11_1 = random.uniform(-0.5, 0.5)
w21_1 = random.uniform(-0.5, 0.5)
w31_1 = random.uniform(-0.5, 0.5)
w41_1 = random.uniform(-0.5, 0.5)
w51_1 = random.uniform(-0.5, 0.5)
w61_1 = random.uniform(-0.5, 0.5)
w71_1 = random.uniform(-0.5, 0.5)
b1_1 = random.uniform(-0.5, 0.5)
b2_1 = random.uniform(-0.5, 0.5)
b3_1 = random.uniform(-0.5, 0.5)
b4_1 = random.uniform(-0.5, 0.5)
b5_1 = random.uniform(-0.5, 0.5)
b6_1 = random.uniform(-0.5, 0.5)
b7_1 = random.uniform(-0.5, 0.5)

#second layer weights
w11_2 = random.uniform(-0.5, 0.5)
w12_2 = random.uniform(-0.5, 0.5)
w13_2 = random.uniform(-0.5, 0.5)
w14_2 = random.uniform(-0.5, 0.5)
w15_2 = random.uniform(-0.5, 0.5)
w16_2 = random.uniform(-0.5, 0.5)
w17_2 = random.uniform(-0.5, 0.5)
b1_2 = random.uniform(-0.5, 0.5)

y_pred = [0]*len(x_vector)
eta = 0.1

for epoch in range(10000):
    for i, x in enumerate(x_vector):
        #first layer weighted sums
        ws1_1 = w11_1 * x + b1_1
        ws2_1 = w21_1 * x + b2_1
        ws3_1 = w31_1 * x + b3_1
        ws4_1 = w41_1 * x + b4_1
        ws5_1 = w51_1 * x + b5_1
        ws6_1 = w61_1 * x + b6_1
        ws7_1 = w71_1 * x + b7_1
        #pasleptojo sluoksnio isejimas
        f1_1 = 1/(1+np.exp(-ws1_1))
        f2_1 = 1/(1+np.exp(-ws2_1))
        f3_1 = 1/(1+np.exp(-ws3_1))
        f4_1 = 1/(1+np.exp(-ws4_1))
        f5_1 = 1/(1+np.exp(-ws5_1))
        f6_1 = 1/(1+np.exp(-ws6_1))
        f7_1 = 1/(1+np.exp(-ws7_1))
        
        #second layer weighted sums
        ws1_2 = w11_2 * f1_1 +\
                w12_2 * f2_1 +\
                w13_2 * f3_1 +\
                w14_2 * f4_1 +\
                w15_2 * f5_1 +\
                w16_2 * f6_1 +\
                w17_2 * f7_1 + b1_2
        # exit layer output
        f1_2 = ws1_2

        # update function
        y_pred[i] = f1_2
        # calculate error
        e = y_exp[i] - f1_2
        delta1_2 = e
        # update weights
        w11_2 += eta * delta1_2 * f1_1
        w12_2 += eta * delta1_2 * f2_1
        w13_2 += eta * delta1_2 * f3_1
        w14_2 += eta * delta1_2 * f4_1
        w15_2 += eta * delta1_2 * f5_1
        w16_2 += eta * delta1_2 * f6_1
        w17_2 += eta * delta1_2 * f7_1
        
        b1_2 += eta * delta1_2

        # error gradient of the hidden layer
        delta1_1 = 1/(1+np.exp(-ws1_1)) * (1-1/(1+np.exp(-ws1_1))) * delta1_2 * w11_2
        delta2_1 = 1/(1+np.exp(-ws2_1)) * (1-1/(1+np.exp(-ws2_1))) * delta1_2 * w12_2
        delta3_1 = 1/(1+np.exp(-ws3_1)) * (1-1/(1+np.exp(-ws3_1))) * delta1_2 * w13_2
        delta4_1 = 1/(1+np.exp(-ws4_1)) * (1-1/(1+np.exp(-ws4_1))) * delta1_2 * w14_2
        delta5_1 = 1/(1+np.exp(-ws5_1)) * (1-1/(1+np.exp(-ws5_1))) * delta1_2 * w15_2
        delta6_1 = 1/(1+np.exp(-ws6_1)) * (1-1/(1+np.exp(-ws6_1))) * delta1_2 * w16_2
        delta7_1 = 1/(1+np.exp(-ws7_1)) * (1-1/(1+np.exp(-ws7_1))) * delta1_2 * w17_2
        
        # update weights in the hidden layer
        w11_1 += eta * delta1_1 * x
        w21_1 += eta * delta2_1 * x
        w31_1 += eta * delta3_1 * x
        w41_1 += eta * delta4_1 * x
        w51_1 += eta * delta5_1 * x
        w61_1 += eta * delta6_1 * x
        w71_1 += eta * delta7_1 * x
        b1_1 += eta * delta1_1
        b2_1 += eta * delta2_1
        b3_1 += eta * delta3_1
        b4_1 += eta * delta4_1
        b5_1 += eta * delta5_1
        b6_1 += eta * delta6_1
        b7_1 += eta * delta7_1

plt.plot(x_vector, y_pred)
plt.show()

# # nauji duomenys: x = 0.1:1/220:1
x_vector = np.arange(0.1, 1.0 + 1/220, 1/220)
y_pred = [0]*len(x_vector)

for x in x_vector:
    #first layer weighted sums
    ws1_1 = w11_1 * x + b1_1
    ws2_1 = w21_1 * x + b2_1
    ws3_1 = w31_1 * x + b3_1
    ws4_1 = w41_1 * x + b4_1
    ws5_1 = w51_1 * x + b5_1
    ws6_1 = w61_1 * x + b6_1
    ws7_1 = w71_1 * x + b7_1
    #pasleptojo sluoksnio isejimas
    f1_1 = 1/(1+np.exp(-ws1_1))
    f2_1 = 1/(1+np.exp(-ws2_1))
    f3_1 = 1/(1+np.exp(-ws3_1))
    f4_1 = 1/(1+np.exp(-ws4_1))
    f5_1 = 1/(1+np.exp(-ws5_1))
    f6_1 = 1/(1+np.exp(-ws6_1))
    f7_1 = 1/(1+np.exp(-ws7_1))

    #second layer weighted sums
    ws1_2 = w11_2 * f1_1 +\
            w12_2 * f2_1 +\
            w13_2 * f3_1 +\
            w14_2 * f4_1 +\
            w15_2 * f5_1 +\
            w16_2 * f6_1 +\
            w17_2 * f7_1 + b1_2
    # exit layer output
    f1_2 = ws1_2
    y_pred[i] = f1_2

plt.plot(x_vector, y_pred)
plt.show()