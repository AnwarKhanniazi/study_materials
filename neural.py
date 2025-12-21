import numpy as np

X = np.array([[0.5, 0.7, 0.2], [0.4, 0.1, 0.5]])
W1 = np.array([[0.2, 0.15], [0.1, 0.2]])
j = np.array([[0.5], [0.8]])
b = np.array([[0.2], [0.1]])
W2 = np.array([0.3, 0.7])
y = [0.6, 0.3, 0.4]
E = 0.0001
rate = 0.5
Iterations = 3
m = 3


# normalized the inputs with zero mean and unit variance
def Normalize(X):
    # find mean in both dimensions x1 and x2 for 3 examples
    # for x1
    values = (X[:1, ])
    sum = np.sum(values)
    U1 = 1. / m * sum
    # for x2
    new = (X[1:3])
    sum = np.sum(new)
    U2 = 1. / m * sum
    U = [[U1], [U2]]
    zeromean = (X - U)
    var1 = np.std(values, ddof=1)
    var2 = np.std(new, ddof=1)
    var = ([[var1], [var2]])
    # step7: execute normalization
    out = zeromean / var
    out = np.round(out, 4)
    return out


def Relu(X):
    return np.maximum(0.0, X)


def feedForward(t, w1, w2, j, b):
    # this function calls two other functions ff for layer 1 and ff for layer2
    zee = np.dot(w1, t)
    zee = np.round(zee, 6)
    # NORMALIZING Z WITH ZERO MEAN AND UNIT VARINACE
    z_normalized = Normalize(zee)
    z_normalized = (z_normalized)
    # SCALING WITH J AND BIASING WITH B
    z_norm1 = z_normalized[:1, ]
    z_hat1 = (j[0][0] * z_norm1) + b[0][0]
    z_norm2 = z_normalized[1:3]
    z_hat2 = (j[1][0] * z_norm2) + b[1][0]

    z_hat1 = np.round(z_hat1, 4)
    z_hat2 = np.round(z_hat2, 4)
    a1 = Relu(z_hat1)
    a2 = Relu(z_hat2)
    a = np.concatenate([a1, a2])
    y_predicted = np.dot(w2, a)
    y_predicted = np.round(y_predicted, 4)

    return y_predicted, a1, a2, z_norm1, z_norm2, z_hat1, z_hat2, zee


def Gradient_descent(xv, W1, W2, j, b, y_hat, a1, a2, Y, alpha, znorm1, znorm2, zhat1, zhat2):
    # 10 parameters to learn
    # w1's
    w1_11 = W1[0][0]
    w1_12 = W1[0][1]
    w1_21 = W1[1][0]
    w1_22 = W1[1][1]
    # w2's
    w2_11 = W2[0]
    w2_12 = W2[1]

    # j
    j1_1 = j[0][0]
    j1_2 = j[1][0]
    # b
    b1_1 = b[0][0]
    b1_2 = b[1][0]

    # derivatives answers obtained from solving functions
    # starting from backward direction
    dJ_by_dy_hat = -2 * (y - y_pred)

    dy_hat_by_dw2_11 = a1

    dy_hat_by_dw2_12 = a2

    dy_hat_by_da1 = w2_11

    dy_hat_by_da2 = w2_12

    da1_by_z_hat1 = z_hat1

    da2_by_z_hat2 = z_hat2

    # wrt relu
    if (zhat1[0][0] >= 0):
        zhat1[0][0] = 1
    if (zhat1[0][0] < 0):
        zhat1[0][0] = 0
    if (zhat1[0][1] >= 0):
        zhat1[0][1] = 1
    if (zhat1[0][1] < 0):
        zhat1[0][1] = 0
    if (zhat1[0][2] >= 0):
        zhat1[0][2] = 1
    if (zhat1[0][2] < 0):
        zhat1[0][2] = 0

        # for zhat2
    if (zhat2[0][0] >= 0):
        zhat2[0][0] = 1
    if (zhat2[0][0] < 0):
        zhat2[0][0] = 0
    if (zhat2[0][1] >= 0):
        zhat2[0][1] = 1
    if (zhat2[0][1] < 0):
        zhat2[0][1] = 0
    if (zhat2[0][2] >= 0):
        zhat2[0][2] = 1
    if (zhat2[0][2] < 0):
        zhat2[0][2] = 0

    dz_hat1_by_dj1_1 = znorm1

    dz_hat2_by_dj1_2 = znorm2

    dz_hat1_by_db1_1 = 1

    dz_hat2_by_db1_2 = 1

    dz_hat1_by_dz_norm1 = j1_1

    dz_hat2_by_dz_norm2 = j1_2

    r = (z[:1, ])
    c = (z[1:3])

    dz_norm1_by_dz1 = np.std(r, ddof=1)
    dz_norm1_by_dz1 = (1 / dz_norm1_by_dz1)

    dz_norm2_by_dz2 = np.std(c, ddof=1)
    dz_norm2_by_dz2 = (1 / dz_norm2_by_dz2)

    dz1_by_dw1_11 = xv[:1, ]

    dz1_by_dw1_12 = xv[1:3]

    dz2_by_dw2_21 = xv[:1, ]

    dz2_by_dw2_22 = xv[1:3]

    # 10 chainRule equations for calculating new values of parameters

    w2_11 = W2[0] - (alpha * np.round(np.sum(dJ_by_dy_hat * dy_hat_by_dw2_11), 4))
    w2_12 = W2[1] - (alpha * np.round(np.sum(dJ_by_dy_hat * dy_hat_by_dw2_12), 4))
    W2[0] = np.round(w2_11, 4)
    W2[1] = np.round(w2_12, 4)
    b1_1 = b1_1 - (alpha * np.round(np.sum(dJ_by_dy_hat * dy_hat_by_da1 * da1_by_z_hat1 * dz_hat1_by_db1_1), 4))
    b1_2 = b1_2 - (alpha * np.round(np.sum(dJ_by_dy_hat * dy_hat_by_da2 * da2_by_z_hat2 * dz_hat2_by_db1_2), 4))
    b[0][0] = np.round(b1_1, 4)
    b[1][0] = np.round(b1_2, 4)
    j1_1 = j1_1 - (alpha * np.round(np.sum(dJ_by_dy_hat * dy_hat_by_da1 * da1_by_z_hat1 * dz_hat1_by_dj1_1), 4))
    j1_2 = j1_2 - (alpha * np.round(np.sum(dJ_by_dy_hat * dy_hat_by_da2 * da2_by_z_hat2 * dz_hat2_by_dj1_2), 4))
    j[0][0] = np.round(j1_1, 4)
    j[1][0] = np.round(j1_2, 4)

    w1_11 = w1_11 - (alpha * np.round(
        np.sum(dJ_by_dy_hat * dy_hat_by_da1 * da1_by_z_hat1 * dz_hat1_by_dz_norm1 * dz_norm1_by_dz1 * dz1_by_dw1_11),
        4))
    w1_12 = w1_12 - (alpha * np.round(
        np.sum(dJ_by_dy_hat * dy_hat_by_da1 * da1_by_z_hat1 * dz_hat1_by_dz_norm1 * dz_norm1_by_dz1 * dz1_by_dw1_12),
        4))
    w1_21 = w1_21 - (alpha * np.round(
        np.sum(dJ_by_dy_hat * dy_hat_by_da2 * da2_by_z_hat2 * dz_hat2_by_dz_norm2 * dz_norm2_by_dz2 * dz2_by_dw2_21),
        4))
    w1_22 = w1_22 - (alpha * np.round(
        np.sum(dJ_by_dy_hat * dy_hat_by_da2 * da2_by_z_hat2 * dz_hat2_by_dz_norm2 * dz_norm2_by_dz2 * dz2_by_dw2_22),
        4))

    W1[0][0] = np.round(w1_11, 4)
    W1[0][1] = np.round(w1_12, 4)
    W1[1][0] = np.round(w1_21, 4)
    W1[1][1] = np.round(w1_22, 4)

    print("New values of parameters")
    print("W2")
    print(W2)
    print("B")
    print(b)
    print("J")
    print(j)
    print("W1")
    print(W1)

    return W1, W2, j, b


# call normalize function
XNorm = Normalize(X)
print("Normalized Inputs ")
print(XNorm)
# Feed Forwading Inputs are parameters
for i in range(Iterations):
    print('Iteration No', i + 1, "started")
    y_pred, a1, a2, z_norm1, z_norm2, z_hat1, z_hat2, z = feedForward(XNorm, W1, W2, j, b);
    y_pred = np.round(y_pred, 4)
    print("Predicted Values of Y")
    print(y_pred)
    loss = np.sum(np.square(np.subtract(y, y_pred)))
    loss = np.round(loss, 4)
    print("loss")
    print(loss)
    W1, W2, J, B = Gradient_descent(XNorm, W1, W2, j, b, y_pred, a1, a2, y, rate, z_norm1, z_norm2, z_hat1, z_hat2)

print("The loss value is increased in second iteration however in the next iteration it goes down." "\n ",
      "In order to minimize it more if we set learning rate at 0.05 and run gradient decent for 30 number of itertaions then loss value will become exactly equal to zero")