"""
A/B Testing Framework for Thai Regulatory AI.

Compare different optimization strategies:
- Similarity thresholds (0.5 vs 0.6 vs 0.7)
- Document counts (3 vs 5 vs 8)
- Character limits (300 vs 400 vs 500)
- Models (Haiku vs Sonnet)

Metrics tracked:
- Token usage (avg, p95, p99)
- Cost per call
- Response quality (via similarity scores)
- Latency
- User satisfaction (if feedback available)

Usage:
    from code.utils.ab_testing import ABTest, Variant
    
    # Define test
    test = ABTest(
        name="similarity_threshold",
        variants=[
            Variant("control", {"threshold": 0.6}),
            Variant("variant_a", {"threshold": 0.7}),
            Variant("variant_b", {"threshold": 0.5})
        ]
    )
    
    # Assign user to variant
    variant = test.get_variant(session_id)
    
    # Use variant config
    threshold = variant.config["threshold"]
    
    # Record metrics
    test.record(session_id, {
        "tokens": 5000,
        "cost": 0.025,
        "latency_ms": 1200
    })
    
    # Get results
    results = test.get_results()
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
import statistics


@dataclass
class Variant:
    """
    A/B test variant configuration.
    
    Attributes:
        name: Variant identifier (e.g., "control", "variant_a")
        config: Configuration parameters for this variant
        weight: Traffic allocation weight (default: equal distribution)
    """
    name: str
    config: Dict[str, Any]
    weight: float = 1.0


@dataclass
class VariantMetrics:
    """Metrics collected for a variant."""
    variant_name: str
    samples: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    token_samples: List[int] = field(default_factory=list)
    cost_samples: List[float] = field(default_factory=list)
    latency_samples: List[float] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    
    def add_sample(self, metrics: Dict[str, Any]):
        """Add a sample measurement."""
        self.samples += 1
        
        if "tokens" in metrics:
            tokens = metrics["tokens"]
            self.total_tokens += tokens
            self.token_samples.append(tokens)
        
        if "cost" in metrics:
            cost = metrics["cost"]
            self.total_cost += cost
            self.cost_samples.append(cost)
        
        if "latency_ms" in metrics:
            latency = metrics["latency_ms"]
            self.total_latency_ms += latency
            self.latency_samples.append(latency)
        
        if "quality_score" in metrics:
            self.quality_scores.append(metrics["quality_score"])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistical summary."""
        stats = {
            "variant": self.variant_name,
            "samples": self.samples,
        }
        
        if self.token_samples:
            stats["tokens"] = {
                "mean": statistics.mean(self.token_samples),
                "median": statistics.median(self.token_samples),
                "p95": self._percentile(self.token_samples, 0.95),
                "p99": self._percentile(self.token_samples, 0.99),
                "min": min(self.token_samples),
                "max": max(self.token_samples),
            }
        
        if self.cost_samples:
            stats["cost"] = {
                "mean": statistics.mean(self.cost_samples),
                "median": statistics.median(self.cost_samples),
                "total": self.total_cost,
            }
        
        if self.latency_samples:
            stats["latency_ms"] = {
                "mean": statistics.mean(self.latency_samples),
                "median": statistics.median(self.latency_samples),
                "p95": self._percentile(self.latency_samples, 0.95),
                "p99": self._percentile(self.latency_samples, 0.99),
            }
        
        if self.quality_scores:
            stats["quality"] = {
                "mean": statistics.mean(self.quality_scores),
                "median": statistics.median(self.quality_scores),
            }
        
        return stats
    
    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p)
        return sorted_data[min(index, len(sorted_data) - 1)]


class ABTest:
    """
    A/B Test manager.
    
    Handles:
    - Variant assignment (consistent hashing)
    - Metrics collection
    - Statistical analysis
    - Results export
    """
    
    def __init__(
        self,
        name: str,
        variants: List[Variant],
        storage_path: Optional[Path] = None
    ):
        """
        Initialize A/B test.
        
        Args:
            name: Test identifier
            variants: List of variants to test
            storage_path: Path to store test data (optional)
        """
        self.name = name
        self.variants = {v.name: v for v in variants}
        self.metrics = {v.name: VariantMetrics(v.name) for v in variants}
        self.storage_path = storage_path or Path(f"ab_test_{name}.json")
        
        # Calculate cumulative weights for assignment
        total_weight = sum(v.weight for v in variants)
        self.cumulative_weights = []
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.weight / total_weight
            self.cumulative_weights.append((variant.name, cumulative))
        
        # Load existing data if available
        self.load()
    
    def get_variant(self, identifier: str) -> Variant:
        """
        Assign identifier to a variant (consistent hashing).
        
        Args:
            identifier: User/session identifier
        
        Returns:
            Assigned variant
        """
        # Hash identifier to [0, 1]
        hash_value = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000.0
        
        # Find variant based on cumulative weights
        for variant_name, cumulative_weight in self.cumulative_weights:
            if normalized <= cumulative_weight:
                return self.variants[variant_name]
        
        # Fallback to last variant
        return self.variants[self.cumulative_weights[-1][0]]
    
    def record(self, identifier: str, metrics: Dict[str, Any]):
        """
        Record metrics for an identifier.
        
        Args:
            identifier: User/session identifier
            metrics: Metrics dictionary (tokens, cost, latency_ms, etc.)
        """
        variant = self.get_variant(identifier)
        self.metrics[variant.name].add_sample(metrics)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get test results with statistical analysis.
        
        Returns:
            Dictionary with results for each variant
        """
        results = {
            "test_name": self.name,
            "variants": {}
        }
        
        for variant_name, metrics in self.metrics.items():
            results["variants"][variant_name] = metrics.get_statistics()
        
        # Add comparison (if we have a control group)
        if "control" in results["variants"]:
            results["comparison"] = self._compare_to_control(results["variants"])
        
        return results
    
    def _compare_to_control(self, variants: Dict[str, Any]) -> Dict[str, Any]:
        """Compare variants to control group."""
        control = variants.get("control", {})
        comparison = {}
        
        for variant_name, variant_stats in variants.items():
            if variant_name == "control":
                continue
            
            comp = {"variant": variant_name}
            
            # Token comparison
            if "tokens" in control and "tokens" in variant_stats:
                control_tokens = control["tokens"]["mean"]
                variant_tokens = variant_stats["tokens"]["mean"]
                if control_tokens > 0:
                    improvement = (control_tokens - variant_tokens) / control_tokens * 100
                    comp["token_improvement"] = f"{improvement:+.1f}%"
            
            # Cost comparison
            if "cost" in control and "cost" in variant_stats:
                control_cost = control["cost"]["mean"]
                variant_cost = variant_stats["cost"]["mean"]
                if control_cost > 0:
                    improvement = (control_cost - variant_cost) / control_cost * 100
                    comp["cost_improvement"] = f"{improvement:+.1f}%"
            
            # Latency comparison
            if "latency_ms" in control and "latency_ms" in variant_stats:
                control_latency = control["latency_ms"]["mean"]
                variant_latency = variant_stats["latency_ms"]["mean"]
                if control_latency > 0:
                    improvement = (control_latency - variant_latency) / control_latency * 100
                    comp["latency_improvement"] = f"{improvement:+.1f}%"
            
            comparison[variant_name] = comp
        
        return comparison
    
    def save(self):
        """Save test data to disk."""
        data = {
            "name": self.name,
            "variants": {
                name: asdict(variant) 
                for name, variant in self.variants.items()
            },
            "metrics": {
                name: asdict(metrics)
                for name, metrics in self.metrics.items()
            },
            "timestamp": time.time()
        }
        
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load test data from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            # Restore metrics
            for name, metrics_data in data.get("metrics", {}).items():
                if name in self.metrics:
                    self.metrics[name] = VariantMetrics(**metrics_data)
        except Exception as e:
            print(f"Warning: Failed to load A/B test data: {e}")
    
    def print_results(self):
        """Print formatted results to console."""
        results = self.get_results()
        
        print(f"\n{'='*60}")
        print(f"A/B Test Results: {self.name}")
        print(f"{'='*60}\n")
        
        for variant_name, stats in results["variants"].items():
            print(f"Variant: {variant_name}")
            print(f"  Samples: {stats['samples']}")
            
            if "tokens" in stats:
                t = stats["tokens"]
                print(f"  Tokens: mean={t['mean']:.0f}, p95={t['p95']:.0f}, p99={t['p99']:.0f}")
            
            if "cost" in stats:
                c = stats["cost"]
                print(f"  Cost: mean=${c['mean']:.4f}, total=${c['total']:.2f}")
            
            if "latency_ms" in stats:
                l = stats["latency_ms"]
                print(f"  Latency: mean={l['mean']:.0f}ms, p95={l['p95']:.0f}ms")
            
            print()
        
        if "comparison" in results:
            print("Comparison to Control:")
            for variant_name, comp in results["comparison"].items():
                print(f"  {variant_name}:")
                if "token_improvement" in comp:
                    print(f"    Tokens: {comp['token_improvement']}")
                if "cost_improvement" in comp:
                    print(f"    Cost: {comp['cost_improvement']}")
                if "latency_improvement" in comp:
                    print(f"    Latency: {comp['latency_improvement']}")
            print()
        
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Create test
    test = ABTest(
        name="similarity_threshold_test",
        variants=[
            Variant("control", {"threshold": 0.6}, weight=1.0),
            Variant("high_threshold", {"threshold": 0.7}, weight=1.0),
            Variant("low_threshold", {"threshold": 0.5}, weight=1.0),
        ]
    )
    
    # Simulate data
    import random
    for i in range(100):
        session_id = f"session_{i}"
        variant = test.get_variant(session_id)
        
        # Simulate metrics (high threshold = fewer tokens, low = more tokens)
        base_tokens = 7000
        if variant.name == "high_threshold":
            tokens = base_tokens - random.randint(500, 1500)
        elif variant.name == "low_threshold":
            tokens = base_tokens + random.randint(500, 1500)
        else:
            tokens = base_tokens + random.randint(-500, 500)
        
        test.record(session_id, {
            "tokens": tokens,
            "cost": tokens * 0.000005,  # Rough estimate
            "latency_ms": random.randint(800, 2000)
        })
    
    # Print results
    test.print_results()
    
    # Save
    test.save()
