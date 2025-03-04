function step2d_applyPAVRRestriction()


    data = load('C:\Users\balan\IdeaProjects\pyblinkers\Devel\step2d_data_input_applyPAVRRestriction.mat');  % Loads blinkComp, blinkPositions

    signalData= data.signalData;
    params= data.params;
    blinkProps= data.blinkProps;
    blinkFits= data.blinkFits;
    [blinkProps, blinkFits] = applyPAVRRestriction(blinkProps, blinkFits, params, signalData);
end