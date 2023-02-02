

def tester(default):
    print(default)

def main():
    list = []

    while True:
        user_input = int(input('Would you like to\n \
        (1)Add or\n(2)Remove items or\n(3)Quit: '))

        if user_input == 1:
            item = input('What will be added?: ')
            list.append(item)
        elif user_input == 2:
            print('There are {} items in the list'.format(len(list)))
            item = int(input('Which item is deleted?: '))
            try:
                list.pop(item)
            except Exception:
                print('Incorrect selection.')
        elif user_input == 3:
            print('The following items remain in the list:')
            for item in list:
                print(item)
            break
        else:
            print('Incorrect selection.')
            continue


if __name__ == "__main__":
    main()
