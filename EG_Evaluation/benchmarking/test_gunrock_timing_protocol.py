#!/usr/bin/env python3

import unittest

from gunrock_timing_protocol import (
    GunrockTimingProtocolError,
    parse_strict_gunrock_timing,
)


PROTOCOL = (
    "Aligned Timing Protocol : "
    "matrix_market_load_to_device_graph_v1;"
    "matrix_market_load_to_complete_host_result_v1;"
    "processing_native_device_interval_v1"
)


class GunrockTimingProtocolTest(unittest.TestCase):
    def test_maintained_multisource_prefers_total_device_interval(self):
        parsed = parse_strict_gunrock_timing(
            "\n".join(
                (
                    "Aligned Construction Time : 12.000000 (ms)",
                    "Aligned E2E Time : 21.000000 (ms)",
                    PROTOCOL,
                    "GPU Elapsed Time : 1.000000 (ms)",
                    "GPU Total Elapsed Time : 7.000000 (ms)",
                )
            )
        )
        self.assertEqual(parsed["construction_seconds"], 0.012)
        self.assertEqual(parsed["processing_seconds"], 0.007)
        self.assertEqual(parsed["e2e_seconds"], 0.021)
        self.assertFalse(parsed["external_cli_wall_used"])

    def test_legacy_average_is_accepted_as_native_processing(self):
        parsed = parse_strict_gunrock_timing(
            "\n".join(
                (
                    "Aligned Construction Time : 8.000000 (ms)",
                    "Aligned E2E Time : 11.000000 (ms)",
                    PROTOCOL,
                    "Run 0 elapsed: 2.000000 ms",
                    "avg. elapsed: 2.500000 ms",
                )
            )
        )
        self.assertEqual(parsed["processing_seconds"], 0.0025)

    def test_missing_construction_is_rejected_without_wall_fallback(self):
        with self.assertRaisesRegex(
            GunrockTimingProtocolError, "Aligned Construction Time"
        ):
            parse_strict_gunrock_timing(
                "\n".join(
                    (
                        "Aligned E2E Time : 11.000000 (ms)",
                        PROTOCOL,
                        "GPU Elapsed Time : 2.000000 (ms)",
                    )
                )
            )

    def test_processing_cannot_exceed_e2e(self):
        with self.assertRaisesRegex(GunrockTimingProtocolError, "exceeds E2E"):
            parse_strict_gunrock_timing(
                "\n".join(
                    (
                        "Aligned Construction Time : 8.000000 (ms)",
                        "Aligned E2E Time : 11.000000 (ms)",
                        PROTOCOL,
                        "GPU Elapsed Time : 12.000000 (ms)",
                    )
                )
            )

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaisesRegex(
            GunrockTimingProtocolError, "unexpected timing protocol"
        ):
            parse_strict_gunrock_timing(
                "\n".join(
                    (
                        "Aligned Construction Time : 8.000000 (ms)",
                        "Aligned E2E Time : 11.000000 (ms)",
                        "Aligned Timing Protocol : old_boundary",
                        "GPU Elapsed Time : 2.000000 (ms)",
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
