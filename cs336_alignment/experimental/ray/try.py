import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.n = 0
    
    def increment(self):
        self.n += 1
        return self.n

@ray.remote
def f(x):
    return x * x

results =  [f.remote(i) for i in range(10)]
print(ray.get(results))

counter = Counter.remote()
print(ray.get(counter.increment.remote()))
print(ray.get(counter.increment.remote()))