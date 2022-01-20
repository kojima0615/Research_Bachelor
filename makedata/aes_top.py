import encryption_round


def main():

    plain = 0x00112233445566778899aabbccddeeff
    key = 0x000102030405060708090a0b0c0d0e0f
    '''
    with open("plain_1000000_v1.csv") as f:
        plain_text = f.readlines()

    with open("key_1000000_v1.csv") as f:
        key1 = f.readlines()
    '''
    
    out,out1 = encryption_round.enc(plain,key)
    print(out1)
    print(hex(out))
    
#print(round1_sboxout)


if __name__ == "__main__":
    main()
