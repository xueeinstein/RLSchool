import parl
@parl.remote_class
class Agent(object):

    def say_hello(self):
        print("Hello World!")

    def sum(self, a, b):
        return a+b

parl.connect('localhost:8010')
agent = Agent()
agent.say_hello()
ans = agent.sum(1, 5)
print(ans)
