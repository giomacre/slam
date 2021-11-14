from functools import partial
from queue import Empty, Queue
from threading import Thread

from decorators import ddict


def create_thread_context():
    thread_context = ddict()
    thread_context |= ddict(
        __terminated__=False,
        terminated=partial(
            getattr,
            thread_context,
            "__terminated__",
        ),
        close=partial(
            setattr,
            thread_context,
            "__terminated__",
            True,
        ),
        threads=[],
    )
    return thread_context


def create_worker(
    target,
    thread_context,
    one_shot=lambda: None,
):
    queue = Queue()

    def register_task(task):
        queue.put(task)
        return queue.join

    def worker_loop(queue):
        one_shot()
        while True:
            try:
                target(*queue.get(timeout=0.015))
                queue.task_done()
            except Empty:
                target()
                pass
            if thread_context.terminated():
                while not queue.empty():
                    queue.get()
                    queue.task_done()
                break

    thread = Thread(
        target=worker_loop,
        args=(queue,),
        daemon=True,
    )
    thread_context.threads += [thread]
    return register_task
