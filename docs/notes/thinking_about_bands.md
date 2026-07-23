## Thinking about bands

In the case of the He5 files 0 is the invalid pixel identifier the mask basically. I wonder if all the bands in the file have the same number of invalid pixels? Will be interesting to see atleast.

Turns out that each band has a different number of invalid pixels which must be taken into account. Using a masking cube or an invalid pixel cube is a much better approach than using seperate band masks and invalid pixel masks. We need to use the same data structure to represent both.

Maybe the mask can also be squashed and added as another channelto provide some extra context to the model (canto this as it would consider entirely invalid bands and destroy the learning fully - I can try a bitwise AND and that would be cool). The loss on the invalid pixels must nonethe less never be learned.

## Getting back to band identification and slotting