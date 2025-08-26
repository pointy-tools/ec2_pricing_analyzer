#!/usr/bin/env python3
"""
EC2 Pricing and Network Performance Analyzer

A modern Python script for analyzing EC2 instance pricing and network performance,
calculating cost-effectiveness ratios for different egress requirements.
"""

import json
import logging
import sys
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from tabulate import tabulate

# Type-only imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mypy_boto3_ec2.client import EC2Client
    from mypy_boto3_ec2.type_defs import DescribeInstanceTypesResponseTypeDef, InstanceTypeInfoTypeDef
    from mypy_boto3_pricing.client import PricingClient
    from mypy_boto3_pricing.type_defs import GetProductsResponseTypeDef

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


class NetworkingInfo(TypedDict):
    """Type definition for simplified networking information."""
    vcpus: int
    baseline: Optional[float]
    burst: str


@dataclass
class InstanceDetails:
    """Data class for EC2 instance details."""
    instance_type: str
    price: float
    price_monthly: float
    vcpus: int
    baseline: Optional[float]
    burst: str
    max_egress: float
    ratio: float


class EC2PricingAnalyzer:
    """EC2 pricing and network performance analyzer."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize the analyzer with AWS clients."""
        self.region: str = region
        try:
            # Pricing API only available in us-east-1
            if TYPE_CHECKING:
                self.pricing_client: "PricingClient" = boto3.client(
                    "pricing", region_name="us-east-1"
                )
                self.ec2_client: "EC2Client" = boto3.client("ec2", region_name=region)
            else:
                self.pricing_client = boto3.client("pricing", region_name="us-east-1")
                self.ec2_client = boto3.client("ec2", region_name=region)
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise

    def get_instance_pricing(
        self, location: str = "US East (N. Virginia)"
    ) -> Dict[str, float]:
        """
        Retrieve current EC2 instance pricing information.

        Args:
            location: AWS region location name

        Returns:
            Dictionary mapping instance types to hourly prices

        Raises:
            ClientError: AWS API client error
            BotoCoreError: AWS SDK core error
        """
        logger.info("Fetching EC2 pricing data...")
        pricing_data: Dict[str, float] = {}

        try:
            paginator = self.pricing_client.get_paginator("get_products")

            for page in paginator.paginate(
                ServiceCode="AmazonEC2",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "location", "Value": location},
                    {
                        "Type": "TERM_MATCH",
                        "Field": "productFamily",
                        "Value": "Compute Instance",
                    },
                ],
            ):
                if TYPE_CHECKING:
                    # Type hint for mypy
                    page_typed: "GetProductsResponseTypeDef" = page
                    price_list: List[str] = page_typed["PriceList"]
                else:
                    price_list = page["PriceList"]

                for price_item in price_list:
                    try:
                        product_data: Dict[str, Any] = json.loads(price_item)

                        # Skip if not compute instance
                        product_family: Optional[str] = (
                            product_data.get("product", {})
                            .get("productFamily", None)
                        )
                        if product_family != "Compute Instance":
                            continue

                        instance_type: Optional[str] = (
                            product_data.get("product", {})
                            .get("attributes", {})
                            .get("instanceType")
                        )
                        if not instance_type:
                            continue

                        # Extract on-demand pricing
                        terms: Dict[str, Any] = product_data.get("terms", {}).get(
                            "OnDemand", {}
                        )
                        if not terms:
                            continue

                        for term_data in terms.values():
                            price_dimensions: Dict[str, Any] = term_data.get(
                                "priceDimensions", {}
                            )
                            for price_dim in price_dimensions.values():
                                price_per_unit: Optional[str] = None
                                if isinstance(price_dim, dict):
                                    price_per_unit = (
                                        price_dim.get("pricePerUnit", {}).get("USD")
                                    )

                                if price_per_unit:
                                    try:
                                        price: float = float(price_per_unit)
                                        if price > 0:
                                            pricing_data[instance_type] = price
                                            break
                                    except (ValueError, TypeError):
                                        continue
                            if instance_type in pricing_data:
                                break

                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning(f"Error processing price item: {e}")
                        continue

            logger.info(f"Retrieved pricing for {len(pricing_data)} instance types")
            return pricing_data

        except (ClientError, BotoCoreError) as e:
            logger.error(f"AWS API error while fetching pricing: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching pricing: {e}")
            raise

    def get_instance_networking_info(self) -> Dict[str, NetworkingInfo]:
        """
        Retrieve EC2 instance networking information.

        Returns:
            Dictionary mapping instance types to networking details

        Raises:
            ClientError: AWS API client error
            BotoCoreError: AWS SDK core error
        """
        logger.info("Fetching EC2 instance networking information...")
        networking_data: Dict[str, NetworkingInfo] = {}

        try:
            paginator = self.ec2_client.get_paginator("describe_instance_types")

            for page in paginator.paginate():
                if TYPE_CHECKING:
                    # Type hint for mypy
                    page_typed: "DescribeInstanceTypesResponseTypeDef" = page
                    instance_types: List["InstanceTypeInfoTypeDef"] = page_typed["InstanceTypes"]
                else:
                    instance_types = page["InstanceTypes"]

                for instance_type_data in instance_types:
                    instance_type: str = instance_type_data["InstanceType"]

                    # Extract networking info safely
                    vcpu_info = instance_type_data.get("VCpuInfo", {})
                    network_info = instance_type_data.get("NetworkInfo", {})
                    network_cards = network_info.get("NetworkCards", [])

                    # Safely extract baseline bandwidth
                    baseline_bandwidth: Optional[float] = None
                    if network_cards and len(network_cards) > 0:
                        first_card = network_cards[0]
                        if "BaselineBandwidthInGbps" in first_card:
                            baseline_bandwidth = first_card["BaselineBandwidthInGbps"]

                    networking_data[instance_type] = NetworkingInfo(
                        vcpus=vcpu_info.get("DefaultVCpus", 0),
                        baseline=baseline_bandwidth,
                        burst=network_info.get("NetworkPerformance", "Unknown"),
                    )

            logger.info(
                f"Retrieved networking info for {len(networking_data)} instance types"
            )
            return networking_data

        except (ClientError, BotoCoreError) as e:
            logger.error(f"AWS API error while fetching networking info: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching networking info: {e}")
            raise

    def merge_and_calculate(
        self,
        pricing_data: Dict[str, float],
        networking_data: Dict[str, NetworkingInfo],
    ) -> Dict[str, InstanceDetails]:
        """
        Merge pricing and networking data and calculate derived fields.

        Args:
            pricing_data: Instance type to price mapping
            networking_data: Instance type to networking info mapping

        Returns:
            Dictionary of merged and calculated instance details
        """
        logger.info("Merging data and calculating derived fields...")
        merged_data: Dict[str, InstanceDetails] = {}

        for instance_type in pricing_data:
            if instance_type in networking_data:
                network_info: NetworkingInfo = networking_data[instance_type]
                baseline: Optional[float] = network_info["baseline"]
                price: float = pricing_data[instance_type]
                vcpus: int = network_info["vcpus"]

                # Skip instances without baseline or with zero price
                if baseline is None or price <= 0:
                    continue

                # Calculate max_egress based on the original logic
                max_egress: float
                if vcpus < 32:
                    max_egress = min(baseline, 5.0)
                else:
                    max_egress = baseline / 2

                # Calculate ratio and monthly price
                ratio: float = max_egress / price if price > 0 else 0.0
                price_monthly: float = price * 730  # Hours in a month

                merged_data[instance_type] = InstanceDetails(
                    instance_type=instance_type,
                    price=price,
                    price_monthly=price_monthly,
                    vcpus=vcpus,
                    baseline=baseline,
                    burst=network_info["burst"],
                    max_egress=max_egress,
                    ratio=ratio,
                )

        logger.info(f"Merged data for {len(merged_data)} instance types")
        return merged_data

    def get_best_instances_by_egress(
        self, instance_data: Dict[str, InstanceDetails]
    ) -> Dict[float, InstanceDetails]:
        """
        Create a dictionary mapping max_egress values to the most cost-effective instance.

        Args:
            instance_data: Merged instance details

        Returns:
            Dictionary mapping max_egress to best instance details
        """
        logger.info("Finding best instances by egress level...")
        egress_to_best_instance: Dict[float, InstanceDetails] = {}

        # Group instances by max_egress and find the one with highest ratio
        egress_groups: Dict[float, List[InstanceDetails]] = {}

        for instance_details in instance_data.values():
            max_egress: float = instance_details.max_egress
            if max_egress not in egress_groups:
                egress_groups[max_egress] = []
            egress_groups[max_egress].append(instance_details)

        # For each egress level, find the instance with the highest ratio
        for max_egress, instances in egress_groups.items():
            best_instance: InstanceDetails = max(instances, key=lambda x: x.ratio if not x.instance_type.startswith("hpc") else float(0))
            egress_to_best_instance[max_egress] = best_instance

        logger.info(
            f"Found best instances for {len(egress_to_best_instance)} egress levels"
        )
        return egress_to_best_instance

    def save_results(
        self,
        data: Union[
            Dict[str, Any],
            Dict[str, InstanceDetails],
            Dict[float, InstanceDetails],
            Dict[str, NetworkingInfo],
        ],
        filename: str,
    ) -> None:
        """
        Save results to JSON file.

        Args:
            data: Data to save (various supported types)
            filename: Output filename

        Raises:
            IOError: File write error
            OSError: Operating system error
        """
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json_data: Dict[str, Any]
                if data and isinstance(list(data.values())[0], InstanceDetails):
                    # Convert InstanceDetails to dict for JSON serialization
                    json_data = {str(k): asdict(v) for k, v in data.items()}
                else:
                    # Convert data to string-keyed dict for JSON compatibility
                    json_data = {str(k): v for k, v in data.items()} if data else {}
                json.dump(json_data, f, indent=2, default=str)
            logger.info(f"Results saved to {filename}")
        except (IOError, OSError) as e:
            logger.error(f"Error saving results to {filename}: {e}")
            raise


def display_table(best_instances: Dict[float, InstanceDetails], ratio_floor: float = 35.0) -> None:
    """
    Display results in tabular format.

    Args:
        best_instances: Dictionary mapping egress levels to best instances
        ratio_floor: Minimum cost/performance ratio to display
    """
    # Filter by ratio floor and sort by max_egress for better readability
    filtered_items = {k: v for k, v in best_instances.items() if v.ratio >= ratio_floor}
    sorted_items: List[tuple[float, InstanceDetails]] = sorted(
        filtered_items.items(), key=lambda x: x[0]
    )

    headers: List[str] = [
        "Max Egress (Gbps)",
        "Instance Type",
        "vCPUs",
        "Baseline (Gbps)",
        "Burst Performance",
        "Hourly Price ($)",
        "Monthly Price ($)",
        "Cost/Performance Ratio",
    ]

    table_data: List[List[str]] = []
    for max_egress, instance in sorted_items:
        baseline_str: str = (
            f"{instance.baseline:.2f}" if instance.baseline else "N/A"
        )
        table_data.append(
            [
                f"{max_egress:.2f}",
                instance.instance_type,
                str(instance.vcpus),
                baseline_str,
                instance.burst,
                f"{instance.price:.4f}",
                f"{instance.price_monthly:.2f}",
                f"{instance.ratio:.2f}",
            ]
        )

    print(f"\nEC2 Instance Cost-Effectiveness by Network Egress Level (Ratio >= {ratio_floor})")
    print("=" * 80)
    if not table_data:
        print(f"No instances found with cost/performance ratio >= {ratio_floor}")
    else:
        print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f"))


def create_argument_parser() -> ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = ArgumentParser(
        description="Analyze EC2 instance pricing and network performance",
        formatter_class=RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--region", default="us-east-1", help="AWS region (default: us-east-1)"
    )

    parser.add_argument(
        "--location",
        default="US East (N. Virginia)",
        help="AWS location for pricing (default: US East (N. Virginia))",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for JSON files (default: current directory)",
    )

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate JSON files (pricing, networking, merged)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress info logging"
    )

    parser.add_argument(
        "--ratio-floor",
        type=float,
        default=20.0,
        help="Minimum cost/performance ratio to display (default: 20.0)",
    )

    return parser


def main() -> None:
    """Main entry point for the CLI."""
    parser: ArgumentParser = create_argument_parser()
    args: Namespace = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Initialize analyzer
        analyzer: EC2PricingAnalyzer = EC2PricingAnalyzer(region=args.region)

        # Fetch data
        pricing_data: Dict[str, float] = analyzer.get_instance_pricing(
            location=args.location
        )
        networking_data: Dict[str, NetworkingInfo] = (
            analyzer.get_instance_networking_info()
        )

        # Save intermediate results if requested
        if args.save_intermediate:
            analyzer.save_results(
                pricing_data, str(args.output_dir / "instance-pricing.json")
            )
            analyzer.save_results(
                networking_data, str(args.output_dir / "instance-networking.json")
            )

        # Merge and calculate
        merged_data: Dict[str, InstanceDetails] = analyzer.merge_and_calculate(
            pricing_data, networking_data
        )

        if args.save_intermediate:
            analyzer.save_results(
                merged_data, str(args.output_dir / "instance-merged.json")
            )

        # Find best instances by egress level
        best_instances: Dict[float, InstanceDetails] = (
            analyzer.get_best_instances_by_egress(merged_data)
        )

        # Save final results
        analyzer.save_results(
            best_instances, str(args.output_dir / "best-instances-by-egress.json")
        )

        # Display results
        display_table(best_instances, args.ratio_floor)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
