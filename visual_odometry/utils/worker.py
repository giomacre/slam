from ctypes import c_bool
from functools import partial

# from multiprocessing.dummy import Condition
from queue import Empty  # , Queue

# from threading import Thread, current_thread
from multiprocessing import Condition
from multiprocessing import Process as Thread, current_process as current_thread
from multiprocessing import JoinableQueue as Queue
from multiprocessing import Value
from .decorators import handle_generator


class ThreadContext:
    def __init__(self):
        self.__close_cond__ = Condition()
        self.__is_closed__ = Value(c_bool, lock=False)
        self.__threads__ = []

    @property
    def is_closed(self):
        with self.__close_cond__:
            return self.__is_closed__.value

    def add_thread(self, thread, signal):
        self.__threads__ += [(thread, signal)]

    def close_context(self):
        with self.__close_cond__:
            self.__is_closed__.value = True
            self.__close_cond__.notify_all()

    def start(self):
        for (thread, _) in self.__threads__:
            thread.start()

    def wait_close(self):
        with self.__close_cond__:
            while not self.is_closed:
                self.__close_cond__.wait()

    def cleanup(self):
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
    timeout=32e-3,
    empty_queue_handler=lambda _: None,
):
    queue = Queue()
    queue_iter = handle_generator(
        iter,
        exception=Empty,
        handler=empty_queue_handler,
    )

    def register_task(*args):
        queue.put(args)
        return lambda: queue.join() if not thread_context.is_closed else None

    def worker_loop(queue):
        one_shot()
        for args in queue_iter(
            partial(queue.get, timeout=timeout),
            None,
        ):
            if thread_context.is_closed:
                break
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
