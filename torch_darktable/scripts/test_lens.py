from collections.abc import Sequence
from lenses import lens
from dataclasses import dataclass


@dataclass(frozen=True)
class Foo:
  a: int
  message: str
  children: tuple['Foo', ...] = ()


foo = Foo(a=1, message='hello', children=(Foo(a=2, message='cruel'), Foo(a=3, message='world', children=())))

modify = lens.children[1].a.modify(lambda x: x + 1000)
print(modify(foo))


set_messages = lens.children.Each().message.set('!!!!!!!!!!!!!!!!!!!!!')
print(set_messages(foo))
