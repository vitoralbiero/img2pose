import time


class TrainLogger(object):
    def __init__(self, batch_size, frequency=50):
        self.batch_size = batch_size
        self.frequency = frequency
        self.init = False
        self.tic = 0
        self.last_batch = 0
        self.running_loss = 0

    def __call__(self, epoch, total_epochs, batch, total, loss):
        if self.last_batch > batch:
            self.init = False
        self.last_batch = batch

        if self.init:
            self.running_loss += loss
            if batch % self.frequency == 0:
                speed = self.frequency * self.batch_size / (time.time() - self.tic)
                self.running_loss = self.running_loss / self.frequency

                log = (
                    f"Epoch: [{epoch + 1}-{total_epochs}] Batch: [{batch}-{total}] "
                    + f"Speed: {speed:.2f} samples/sec Loss: {self.running_loss:.5f}"
                )
                print(log)

                self.running_loss = 0
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()
