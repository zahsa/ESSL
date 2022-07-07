# if adaptive_pb:
def halving():
    # drop_rate = 0.5, gen_drop = 3
    if not g % 3 and g != 0:
        mutpb /= 2
def generational():
    cxpb = 1 - ((g + 1) / num_generations)
    mutpb = ((g + 1) / num_generations)


# elif adaptive_pb == "AGA":
#
# else:
#     raise ValueError(f"invalid adaptive_pb value: {adaptive_pb}")
