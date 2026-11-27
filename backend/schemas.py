from pydantic import BaseModel, ConfigDict, Field

class Chat(BaseModel):
    role: str
    content: str


class ChatHistory(BaseModel):
    chats: list[Chat]


class ProductLookupArgs(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    product_id: int


class OrderLookupArgs(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    order_id: int


class PrepareOrderRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    product_id: int = Field(gt=0)
    quantity: int = Field(gt=0, le=10)