function step3_selectChannels()

    data = load('C:\Users\balan\IdeaProjects\pyblinkers\Devel\step3a_input_selectChannel_compact.mat');  % Loads blinkComp, blinkPositions
    signalData= data.signalData; 
    params = data.params;


    blinks = processBlinkSignalsCompact(signalData, params);
    data_output = load('C:\Users\balan\IdeaProjects\pyblinkers\Devel\step3a_input_selectChannel.mat');
    blinks_output=data_output.blinks;
    [areStructsEqual, diffDetails] = compareblinkpropertiesstructure(blinks.signalData, blinks_output.signalData)
    h=2
end