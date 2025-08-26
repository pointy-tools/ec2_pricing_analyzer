# EC2 pricing analyzer

This is based off the hard work of the [fck-nat](https://github.com/AndrewGuenther/fck-nat/tree/main/docs/pricing_analysis) folks.

This code was largely written by a robot under my direction, it did ok, I did not care enough about this to hand craft it or make sure every line is how I would _prefer_ it.

This thing

- Merges pricing information with EC2 instance details
- calculates max egress per instance type
- buckets instance types by egress
- calculates a cost/performance ratio per instance type
- filters out non-viable instance types (hpc)
- selects the most cost-effective instance type per egress bucket
- filters out buckets below a certain cost ratio

## installation

``` shell
python3 -m venv venv && source venv/bin/activate
python -m pip install -r requirements.txt
```

## usage

basic usage:
``` shell
./ec2_pricing_analyzer.py
```

advanced usage:
``` shell
./ec2_pricing_analyzer.py --help
usage: ec2_pricing_analyzer.py [-h] [--region REGION] [--location LOCATION] [--output-dir OUTPUT_DIR] [--save-intermediate] [--quiet] [--ratio-floor RATIO_FLOOR]

Analyze EC2 instance pricing and network performance

options:
  -h, --help            show this help message and exit
  --region REGION       AWS region (default: us-east-1)
  --location LOCATION   AWS location for pricing (default: US East (N. Virginia))
  --output-dir OUTPUT_DIR
                        Output directory for JSON files (default: current directory)
  --save-intermediate   Save intermediate JSON files (pricing, networking, merged)
  --quiet, -q           Suppress info logging
  --ratio-floor RATIO_FLOOR
                        Minimum cost/performance ratio to display (default: 20.0)
```

example:
``` shell
bash-3.2$ ./ec2_pricing_analyzer.py -q
EC2 Instance Cost-Effectiveness by Network Egress Level (Ratio >= 20.0)
================================================================================
+---------------------+-----------------+---------+-------------------+---------------------+--------------------+---------------------+--------------------------+
|   Max Egress (Gbps) | Instance Type   |   vCPUs |   Baseline (Gbps) | Burst Performance   |   Hourly Price ($) |   Monthly Price ($) |   Cost/Performance Ratio |
+=====================+=================+=========+===================+=====================+====================+=====================+==========================+
|                1.00 | c3.2xlarge      |       8 |              1.00 | High                |               0.05 |               36.50 |                    20.00 |
+---------------------+-----------------+---------+-------------------+---------------------+--------------------+---------------------+--------------------------+
|                1.60 | c6gn.medium     |       1 |              1.60 | Up to 16 Gigabit    |               0.04 |               32.81 |                    35.60 |
+---------------------+-----------------+---------+-------------------+---------------------+--------------------+---------------------+--------------------------+
|                3.12 | c7gn.medium     |       1 |              3.12 | Up to 25 Gigabit    |               0.07 |               48.25 |                    47.28 |
+---------------------+-----------------+---------+-------------------+---------------------+--------------------+---------------------+--------------------------+
|                5.00 | c8gn.large      |       2 |              6.25 | Up to 30 Gigabit    |               0.13 |               91.69 |                    39.81 |
+---------------------+-----------------+---------+-------------------+---------------------+--------------------+---------------------+--------------------------+
|               50.00 | c7gn.8xlarge    |      32 |            100.00 | 100 Gigabit         |               2.00 |             1457.66 |                    25.04 |
+---------------------+-----------------+---------+-------------------+---------------------+--------------------+---------------------+--------------------------+
|               75.00 | c7gn.12xlarge   |      48 |            150.00 | 150 Gigabit         |               3.00 |             2186.50 |                    25.04 |
+---------------------+-----------------+---------+-------------------+---------------------+--------------------+---------------------+--------------------------+
|              100.00 | c7gn.16xlarge   |      64 |            200.00 | 200 Gigabit         |               4.11 |             2997.09 |                    24.36 |
+---------------------+-----------------+---------+-------------------+---------------------+--------------------+---------------------+--------------------------+
|              150.00 | c8gn.24xlarge   |      96 |            300.00 | 300 Gigabit         |               5.86 |             4274.88 |                    25.61 |
+---------------------+-----------------+---------+-------------------+---------------------+--------------------+---------------------+--------------------------+
|              200.00 | g6e.48xlarge    |     192 |            400.00 | 400 Gigabit         |               8.83 |             6447.36 |                    22.64 |
+---------------------+-----------------+---------+-------------------+---------------------+--------------------+---------------------+--------------------------+
```
