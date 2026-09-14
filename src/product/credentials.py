"""Windows 凭据管理器中的远端模型密钥。"""


SERVICE = "ObsidianRAG"
DEEPSEEK_USER = "deepseek-api-key"


class CredentialStore:
    def _keyring(self):
        try:
            import keyring
            return keyring
        except ImportError as error:
            raise RuntimeError("产品缺少凭据存储组件") from error

    def has_deepseek(self):
        try:
            return bool(self._keyring().get_password(SERVICE, DEEPSEEK_USER))
        except Exception:
            return False

    def get_deepseek(self):
        return self._keyring().get_password(SERVICE, DEEPSEEK_USER)

    def set_deepseek(self, value: str):
        value = value.strip()
        if len(value) < 8:
            raise ValueError("API Key 格式无效")
        self._keyring().set_password(SERVICE, DEEPSEEK_USER, value)

    def delete_deepseek(self):
        try:
            self._keyring().delete_password(SERVICE, DEEPSEEK_USER)
        except Exception:
            pass


class MemoryCredentialStore:
    """测试使用，不触碰系统凭据。"""
    def __init__(self): self.value = None
    def has_deepseek(self): return bool(self.value)
    def get_deepseek(self): return self.value
    def set_deepseek(self, value):
        if len(value.strip()) < 8: raise ValueError("API Key 格式无效")
        self.value = value.strip()
    def delete_deepseek(self): self.value = None
