from enum import Enum


class ResponseCodeEnum(Enum):

    def __new__(cls, code: int, desc: str, eng_desc: str):
        obj = object.__new__(cls)
        obj.code = code
        obj.desc = desc
        obj.eng_desc = eng_desc
        return obj

    # 2000~2499
    ACCOUNT_NOT_EXISTED = (2000, "账号不存在", "account not existed")
    PASSWORD_ERROR = (2001, "密码错误", "password error")
