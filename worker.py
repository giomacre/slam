from functools import partial
from queue import Empty, Queue
from threading import Thread, current_thread
from decorators import handle_generator


class ThreadContext:
    def __init__(self):
        self.__is_closed__ = False
        self.__threads__ = []

    @property
    def is_closed(self):
        return self.__is_closed__

    def add_thread(self, thread, signal):
        self.__threads__ += [(thread, signal)]

    def close_context(self):
        self.__is_closed__ = True

    def start(self):
        for (thread, _) in self.__threads__:
            thread.start()

    def terminate_all(self):
        for (_, close_signal) in self.__threads__:
            close_signal()

    def join_all(self):
        for (thread, _) in self.__threads__:
            thread.join()


def create_thread_context():
    return ThreadContext()


def create_worker(
    target,
    thread_context,
    one_shot=lambda: None,
    name=None,
):
    queue = Queue()
    queue_iter = handle_generator(
        iter,
        exception=Empty,
        handler=lambda _: target(),
    )

    def register_task(task):
        queue.put(task)
        return queue.join

    def worker_loop(queue):
        one_shot()
        for args in queue_iter(
            partial(queue.get, timeout=0.015),
            None,
        ):
            target(*args)
            queue.task_done()
        while not queue.empty():
            queue.get()
            queue.task_done()
        print(f"{current_thread()} exiting.")

    thread = Thread(
        target=worker_loop,
        name=name,
        args=(queue,),
        daemon=False,
    )
    thread_context.add_thread(
        thread,
        partial(
            queue.put,
            None,
        ),
    )
    return register_task
