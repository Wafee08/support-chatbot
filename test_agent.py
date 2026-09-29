import sys
sys.path.insert(0, '..')

from agent import support_chat

def test_faq():
    result = support_chat("test", "How do I reset password?")
    assert "response" in result
    assert len(result["response"]) > 0

if __name__ == "__main__":
    test_faq()
    print("✅ Test passed")