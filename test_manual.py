#!/usr/bin/env python3
"""
Manual test script for llm-bedrock plugin.

This script demonstrates all the functionality without requiring AWS credentials.
"""
import sys


def test_model_discovery():
    """Test that models are discovered and registered."""
    print("=" * 70)
    print("TEST 1: Model Discovery")
    print("=" * 70)

    import llm

    models = [m for m in llm.get_models() if m.model_id.startswith("bedrock/")]
    print(f"✅ Discovered {len(models)} Bedrock models")

    # Show first 10 models
    print("\nFirst 10 models:")
    for model in models[:10]:
        print(f"  • {model.model_id}")
        print(f"    Provider: {model.provider if hasattr(model, 'provider') else 'Unknown'}")
        print(f"    Can stream: {model.can_stream}")

    return True


def test_model_id_simplification():
    """Test model ID simplification logic."""
    print("\n" + "=" * 70)
    print("TEST 2: Model ID Simplification")
    print("=" * 70)

    from llm_bedrock import simplify_model_id

    test_cases = [
        ("anthropic.claude-3-sonnet-20240229-v1:0", "claude-3-sonnet"),
        ("amazon.nova-pro-v1:0", "nova-pro"),
        ("meta.llama3-70b-instruct-v1:0", "llama3-70b-instruct"),
        ("mistral.mistral-large-2407-v1:0", "mistral-large"),
    ]

    all_passed = True
    for bedrock_id, expected in test_cases:
        result = simplify_model_id(bedrock_id)
        passed = result == expected
        all_passed = all_passed and passed
        status = "✅" if passed else "❌"
        print(f"{status} {bedrock_id}")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")

    return all_passed


def test_model_attributes():
    """Test that models have correct attributes."""
    print("\n" + "=" * 70)
    print("TEST 3: Model Attributes")
    print("=" * 70)

    import llm

    # Find a Claude model
    claude_models = [
        m for m in llm.get_models()
        if m.model_id.startswith("bedrock/claude")
    ]

    if not claude_models:
        print("❌ No Claude models found")
        return False

    model = claude_models[0]
    print(f"Testing model: {model.model_id}")

    checks = [
        ("Has model_id", hasattr(model, "model_id")),
        ("Has bedrock_model_id", hasattr(model, "bedrock_model_id")),
        ("Has region", hasattr(model, "region")),
        ("Can stream", model.can_stream),
        ("Has attachment_types", hasattr(model, "attachment_types")),
        ("Supports images", "image/jpeg" in model.attachment_types),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        all_passed = all_passed and result

    return all_passed


def test_converse_compatibility():
    """Test Converse API compatibility detection."""
    print("\n" + "=" * 70)
    print("TEST 4: Converse Compatibility Detection")
    print("=" * 70)

    from llm_bedrock import is_converse_compatible

    compatible = [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "amazon.nova-pro-v1:0",
        "meta.llama3-70b-instruct-v1:0",
    ]

    incompatible = [
        "ai21.jamba-1-5-mini-v1:0",
        "stability.stable-diffusion-xl-v1:0",
    ]

    all_passed = True

    print("Should be compatible:")
    for model_id in compatible:
        result = is_converse_compatible({"modelId": model_id})
        status = "✅" if result else "❌"
        print(f"{status} {model_id}")
        all_passed = all_passed and result

    print("\nShould NOT be compatible:")
    for model_id in incompatible:
        result = is_converse_compatible({"modelId": model_id})
        status = "✅" if not result else "❌"
        print(f"{status} {model_id}")
        all_passed = all_passed and not result

    return all_passed


def test_cache_system():
    """Test model cache functionality."""
    print("\n" + "=" * 70)
    print("TEST 5: Cache System")
    print("=" * 70)

    from llm_bedrock import ModelCache
    import llm
    from pathlib import Path

    cache = ModelCache()
    cache_dir = Path(llm.user_dir()) / "bedrock-models"

    checks = [
        ("Cache directory exists", cache_dir.exists()),
        ("Cache directory is writable", cache_dir.is_dir()),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        all_passed = all_passed and result

    # Check if cache file exists for default region
    cache_file = cache.get_cache_path("us-east-1")
    if cache_file.exists():
        print(f"✅ Cache file exists: {cache_file}")
        import json
        with open(cache_file) as f:
            data = json.load(f)
        print(f"   Timestamp: {data.get('timestamp')}")
        print(f"   Models cached: {len(data.get('models', []))}")
    else:
        print(f"ℹ️  No cache file yet: {cache_file}")

    return all_passed


def test_plugin_integration():
    """Test plugin is properly integrated with LLM."""
    print("\n" + "=" * 70)
    print("TEST 6: Plugin Integration")
    print("=" * 70)

    import llm

    # Check plugin is loaded
    plugins = llm.get_plugins()
    bedrock_plugin = None
    for plugin in plugins:
        if plugin["name"] == "llm-bedrock":
            bedrock_plugin = plugin
            break

    if not bedrock_plugin:
        print("❌ Plugin not loaded")
        return False

    print("✅ Plugin loaded")
    print(f"   Name: {bedrock_plugin['name']}")
    print(f"   Version: {bedrock_plugin['version']}")
    print(f"   Hooks: {', '.join(bedrock_plugin['hooks'])}")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LLM-BEDROCK PLUGIN TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        test_model_discovery,
        test_model_id_simplification,
        test_model_attributes,
        test_converse_compatibility,
        test_cache_system,
        test_plugin_integration,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n❌ {test.__name__} raised an exception:")
            print(f"   {type(e).__name__}: {e}")
            results.append((test.__name__, False))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
