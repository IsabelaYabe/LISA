import sys
print(f"sys.path: {sys.path}")
try:
    from ..import foo
    print("Conseguimos importar foo :)")
except:
    print("Não conseguimos importar foo :(")

sys.path.append(".")
print(f"after sys.path.append('.'): {sys.path}")
try:
    from ..import foo
    print("Conseguimos importar foo :)")
except:
    print("Não conseguimos importar foo :(")

sys.path.append("..")
print(f"after sys.path.append('..'): {sys.path}")
try:
    from ..import foo
    print("Conseguimos importar foo :)")
except:
    print("Não conseguimos importar foo :(")

#import egg


#def bar_func():
#    print("bar")
#    
#if __name__ == "__main__":
#    egg.egg_func()