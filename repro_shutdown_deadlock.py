import ray


@ray.remote
def ping():
    return "pong"


@ray.remote
class Echo:
    def ping(self):
        return "pong"


def main():
    print("[repro] Initializing Ray")
    ray.init()
    print("[repro] Task result:", ray.get(ping.remote()))
    a = Echo.remote()
    print("[repro] Actor result:", ray.get(a.ping.remote()))
    print("[repro] Calling ray.shutdown() — expected to hang due to injected deadlock")
    ray.shutdown()
    print("[repro] This line should never print if deadlock reproduces")


if __name__ == "__main__":
    main()


