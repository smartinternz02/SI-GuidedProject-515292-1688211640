import queue
stack=queue.LifoQueue()
s=input()
for i in s:
    stack.put(i)
while stack:
    print(stack.get(),end="")
