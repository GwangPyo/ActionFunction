

if __name__ == "__main__":
    aae = AAE(shape=(50, 160), channel=3)
    aae.load()
    AtariEmbedding(aae)
    env = AtariMulti(target_game=gym.make("Breakout-v0"), num_seq=5, neuro_structure=(4, 4))


