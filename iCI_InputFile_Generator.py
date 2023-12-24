# coding=UTF-8

iCI_Keywords = [
    "irrep",
    "unpair",
    "nstates",
    "nvalelec",
    "etol",
    "cmin",
    "rotatemo",
    "perturbation",
    "nsegment",
    "spin",
    "inputcfg",
    "epsilon2",
    "davidson",
    "direct",
    "task",
    "dumprdm",
    "relative",
    "ewin_ini",
    "selection",
    "print",
    "doublegroup",
    "prune"]


def _Generate_InputFile_iCI(inputfilename,
                            Segment,
                            nelec_val,
                            rotatemo,
                            cmin,
                            perturbation,
                            dumprdm,
                            relative,
                            Task,
                            inputocfg,
                            etol,
                            selection=1,
                            doublegroup=None,
                            direct=None):
    inputfile = open(inputfilename, "w")
    inputfile.write("nsegment=%s\n" % (Segment))
    inputfile.write(
        "nvalelec=%d\nETOL=%e\nCMIN=%s\nROTATEMO=%d\n" %
        (nelec_val, etol, cmin, rotatemo))
    inputfile.write("perturbation=%d 0\ndumprdm=%d\nrelative=%d\n" %
                    (perturbation, dumprdm, relative))
    inputfile.write("task=%s\n" % (Task))
    inputfile.write("inputcfg=%s\n" % (inputocfg))
    inputfile.write("print=11\n")
    inputfile.write("selection=%s\n" % (selection))
    if doublegroup is not None:
        inputfile.write("doublegroup=%s\n" % (doublegroup))
    if direct is not None:
        inputfile.write("direct=%d\n" % (direct))
    inputfile.close()


def _generate_task_spinarray_weight(state):
    res = ""
    spinarray = []
    weight = []
    for i in range(len(state)):
        res += str(state[i][0])+" "  # spintwo
        res += str(state[i][1])+" "  # irrepID
        res += str(state[i][2])+" "  # nstates
        if (len(state[i]) == 4):     # weight
            for j in state[i][3]:
                res += str(j)+" "
                weight.append(float(j))
                spinarray.append(state[i][0])
        else:
            nstates = state[i][2]
            for j in range(nstates):
                res += "1 "
                weight.append(1.0)
                spinarray.append(state[i][0])

    total = 0.0
    for x in weight:
        total += x
    weight = [x/total for x in weight]

    return res[:-1], spinarray, weight  # 最后一格空格不能要
