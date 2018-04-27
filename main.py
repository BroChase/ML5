import img_load
import tens_flow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    f = open('report.doc', 'w')
    path = 'fingerspelling5/dataset5/A'
    alpha = .001
    max_epochs = 10
    batch_size = 256
    n_classes = 25
    resized_image = (32, 32)
    print('Loading Images from ', path)
    dataset = img_load.get_images(path, n_classes, resized_image)
    print(dataset.X.shape[0], ' Images Loaded')

    idx_train, idx_test = train_test_split(range(dataset.X.shape[0]), test_size=0.20)
    X_train = dataset.X[idx_train, :, :, :]
    X_test = dataset.X[idx_test, :, :, :]
    y_train = dataset.y[idx_train, :]
    y_test = dataset.y[idx_test, :]

    print('X_train shape ', X_train.shape)
    print('y_train shape ', y_train.shape)
    print('X_test shape ', X_test.shape)
    print('y_test shape ', y_test.shape)

    tens_flow.train_model(X_train, y_train, X_test, y_test, alpha, max_epochs, batch_size, resized_image, n_classes, f)
    print('done')
    f.close()


