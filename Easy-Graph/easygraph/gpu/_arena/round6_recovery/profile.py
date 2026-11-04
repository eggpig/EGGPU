"""Charged bounded startup calibration and separately frozen R5 training data."""
import time
import numpy as np
import cupy as cp


def calibrate(session, historical):
    started = time.perf_counter()
    a = session.arena.slots['@native_bfs_output'].array
    b = session.arena.slots['@x'].array
    size = min(a.nbytes, b.nbytes, 1 << 20)
    sizes = sorted(set((max(8, size // 4), size)))
    samples = []
    for n in sizes:
        host = np.zeros(n, np.uint8)
        target = cp.ndarray((n,), dtype=np.uint8, memptr=a.data)
        target.set(host)
        for repeat in range(5):
            t = time.perf_counter()
            target.set(host)
            cp.cuda.runtime.deviceSynchronize()
            h2d = time.perf_counter() - t
            t = time.perf_counter()
            cp.cuda.runtime.memcpyAsync(b.data.ptr, a.data.ptr, n, 3, cp.cuda.get_current_stream().ptr)
            cp.cuda.runtime.deviceSynchronize()
            samples.append(dict(bytes=n, repeat=repeat, h2d_seconds=h2d,
                                d2d_seconds=time.perf_counter()-t))
    def fit(field):
        means = [float(np.median([s[field] for s in samples if s['bytes']==n])) for n in sizes]
        slope = max(0., (means[-1]-means[0])/max(1, sizes[-1]-sizes[0]))
        return slope, max(0., means[0]-sizes[0]*slope)
    h2d, upload = fit('h2d_seconds'); d2d, copy_launch = fit('d2d_seconds')
    launches = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(20):
            session.executor.kernels['init']((1,), (128,),
                (np.int32(1), session.executor.x, np.int32(0), np.int32(0)))
        cp.cuda.runtime.deviceSynchronize(); launches.append((time.perf_counter()-t)/20)
    return dict(h2d_per_byte=h2d, d2d_per_byte=d2d,
                restore_fixed=upload, packet_fixed=upload+historical['packet_seconds_median'],
                kernel_launch_seconds=float(np.median(launches)), copy_launch_seconds=copy_launch,
                samples=samples, kernel_launch_samples=launches, historical=historical,
                calibration_operations_per_sample=1, kernel_operations_per_sample=20,
                calibration_h2d_bytes=sum(sizes)+sum(s['bytes'] for s in samples),
                calibration_d2d_payload_bytes=sum(s['bytes'] for s in samples),
                calibration_d2d_read_write_bytes=2*sum(s['bytes'] for s in samples),
                calibration_seconds=time.perf_counter()-started,
                objective='estimated transfer and maintenance service time; end-to-end time measured separately',
                scope='borrows existing charged workspace; no private device allocations')
