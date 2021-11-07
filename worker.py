from functools import partial
from multiprocessing import Process, Queue


def create_worker_process(
    target,
    terminate=lambda: False,
    one_shot=lambda: None,
):
    queue = Queue()

    def worker_loop(queue):
        one_shot()
        while True:
            target(*queue.get(timeout=1))
            if terminate():
                break

    process = Process(
        target=worker_loop,
        args=(queue,),
        daemon=True,
    )
    process.start()
    return partial(queue.put, block=False)
