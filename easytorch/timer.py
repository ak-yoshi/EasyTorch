import time

#
# class to measure elapsed time
# usage : measure time as follows (ex : with Timer.Timer() as timer:)
class Timer:
  def __enter__(self):
    self.start_time = time.time()
  def __exit__(self, exc_type, exc_value, traceback):
    self.end_time = time.time()
    print("time : {:.3f}sec".format(self.end_time-self.start_time))