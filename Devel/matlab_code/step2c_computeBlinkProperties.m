function step2c_computeBlinkProperties()
    data_output = load('C:\Users\balan\IdeaProjects\pyblinkers\Devel\step2c_data_output_computeBlinkProperties.mat');  % Loads blinkComp, blinkPositions
    
    blinkProps_output=data_output.blinkProps;
    
    data = load('C:\Users\balan\IdeaProjects\pyblinkers\Devel\step2c_data_input_computeBlinkProperties.mat');  % Loads blinkComp, blinkPositions
    blinkFits= data.blinkFits;
    signalData= data.signalData;
    params= data.params;
    srate = data.srate;
    blinkVelocity= data.blinkVelocity;
    peaks= data.peaks;


    [blinkProps, peaksPosVelZero, peaksPosVelBase] = computeBlinkProperties(blinkFits, signalData, params, srate, blinkVelocity, peaks);


    peaksPosVelZero_output=data_output.peaksPosVelZero;
    peaksPosVelBase_output=data_output.peaksPosVelBase;

    [areStructsEqual_blinkProps, diffDetails_blinkProps] = compareblinkpropertiesstructure(blinkProps, blinkProps_output)
    result_peaksPosVelZero = compare_matrices(peaksPosVelZero , peaksPosVelZero_output);
    result_peaksPosVelBase= compare_matrices(peaksPosVelBase , peaksPosVelBase_output);
    h=2
    % elementwise_comparison_peaksPosVelZero = (peaksPosVelZero == peaksPosVelZero_output);
    % elementwise_comparison_peaksPosVelBase = (peaksPosVelBase == peaksPosVelBase_output);

    h=1
end
