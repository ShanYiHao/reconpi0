"""
Simplified array_typing.py compatible with latest jaxtyping.
Includes specific bit-width types required by image_tools.
"""
from typing import Any

# 引入核心类型，补充 UInt8, Float32, Int32 等具体类型
from jaxtyping import (
    Array, 
    Bool, 
    Complex, 
    Float, 
    Inexact, 
    Int, 
    Integer, 
    Num, 
    Shaped,
    UInt8,   # <--- 补上了这个
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32, # <--- 补上了这个
    Float64,
    Int8,
    Int16,
    Int32,   # <--- 补上了这个
    Int64,
    jaxtyped
)
from beartype import beartype

def typecheck(cls_or_func: Any) -> Any:
    """
    Decorator to enable runtime type checking using jaxtyping + beartype.
    """
    return jaxtyped(typechecker=beartype)(cls_or_func)

# 导出所有常用类型
__all__ = [
    "Array",
    "Bool",
    "Complex",
    "Float",
    "Inexact",
    "Int",
    "Integer",
    "Num",
    "Shaped",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float16", "Float32", "Float64",
    "Int8", "Int16", "Int32", "Int64",
    "typecheck",
]