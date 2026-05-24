from dataclasses import dataclass, field

@dataclass
class MenuItem:
    name: str
    count: str
    price: float

@dataclass  
class ReceiptSchema:
    menu: list[MenuItem] = field(default_factory=list)
    tax: float | None = None
    total: float | None = None