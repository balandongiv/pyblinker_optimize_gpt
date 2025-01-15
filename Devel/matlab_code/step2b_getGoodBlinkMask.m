function step2b_getGoodBlinkMask()
    data = load('C:\Users\balan\IdeaProjects\pyblinkers\Devel\step2b_data_input_getGoodBlinkMask.mat');  % Loads blinkComp, blinkPositions
    zThresholds= data.zThresholds; 
    specifiedStd = data.specifiedStd;
    specifiedMedian= data.specifiedMedian;
    blinkFits= data.blinkFits;
    [goodBlinkMask, specifiedMedian, specifiedStd] = ...
      getGoodBlinkMask(blinkFits, specifiedMedian, specifiedStd, zThresholds);

    data_output = load('C:\Users\balan\IdeaProjects\pyblinkers\Devel\step2b_data_output_getGoodBlinkMask.mat');
    
    goodBlinkMask_output=data_output.goodBlinkMask;
    specifiedMedian_output=data_output.specifiedMedian;
    specifiedStd_output=data_output.specifiedStd;

    % [areStructsEqual, diffDetails] = compareblinkpropertiesstructure(struct1, struct2)
    findingx=isequal(goodBlinkMask_output,goodBlinkMask) % Return True 1 if same
    g=1
end