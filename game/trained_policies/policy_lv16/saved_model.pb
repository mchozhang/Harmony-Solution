ķ
Ł
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ņ
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
¬
(QNetwork/EncodingNetwork/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: d*9
shared_name*(QNetwork/EncodingNetwork/dense_20/kernel
„
<QNetwork/EncodingNetwork/dense_20/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_20/kernel*
_output_shapes

: d*
dtype0
¤
&QNetwork/EncodingNetwork/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*7
shared_name(&QNetwork/EncodingNetwork/dense_20/bias

:QNetwork/EncodingNetwork/dense_20/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_20/bias*
_output_shapes
:d*
dtype0

QNetwork/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d0*)
shared_nameQNetwork/dense_21/kernel

,QNetwork/dense_21/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_21/kernel*
_output_shapes

:d0*
dtype0

QNetwork/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_nameQNetwork/dense_21/bias
}
*QNetwork/dense_21/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_21/bias*
_output_shapes
:0*
dtype0

NoOpNoOp
Õ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bü
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
	3


0
 
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_20/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_20/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEQNetwork/dense_21/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEQNetwork/dense_21/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE

ref
1


_q_network
t
_encoder
_q_value_layer
regularization_losses
trainable_variables
	variables
	keras_api
n
_postprocessing_layers
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
 

0
1
2
	3

0
1
2
	3
­
layer_metrics
metrics

layers
layer_regularization_losses
 non_trainable_variables
regularization_losses
trainable_variables
	variables

!0
"1
 

0
1

0
1
­
#layer_metrics
$metrics

%layers
&layer_regularization_losses
'non_trainable_variables
regularization_losses
trainable_variables
	variables
 

0
	1

0
	1
­
(layer_metrics
)metrics

*layers
+layer_regularization_losses
,non_trainable_variables
regularization_losses
trainable_variables
	variables
 
 

0
1
 
 
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

kernel
bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
 
 

!0
"1
 
 
 
 
 
 
 
 
 
 
­
5layer_metrics
6metrics

7layers
8layer_regularization_losses
9non_trainable_variables
-regularization_losses
.trainable_variables
/	variables
 

0
1

0
1
­
:layer_metrics
;metrics

<layers
=layer_regularization_losses
>non_trainable_variables
1regularization_losses
2trainable_variables
3	variables
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

action_0/observation/0Placeholder*/
_output_shapes
:’’’’’’’’’*
dtype0*$
shape:’’’’’’’’’
y
action_0/observation/1Placeholder*'
_output_shapes
:’’’’’’’’’0*
dtype0*
shape:’’’’’’’’’0
j
action_0/rewardPlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
m
action_0/step_typePlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observation/0action_0/observation/1action_0/rewardaction_0/step_type(QNetwork/EncodingNetwork/dense_20/kernel&QNetwork/EncodingNetwork/dense_20/biasQNetwork/dense_21/kernelQNetwork/dense_21/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1559699
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ś
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1559711
Ū
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1559733

StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1559726
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_20/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_20/bias/Read/ReadVariableOp,QNetwork/dense_21/kernel/Read/ReadVariableOp*QNetwork/dense_21/bias/Read/ReadVariableOpConst*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_1559833
Ŗ
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable(QNetwork/EncodingNetwork/dense_20/kernel&QNetwork/EncodingNetwork/dense_20/biasQNetwork/dense_21/kernelQNetwork/dense_21/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_1559858¹ä
ų

 __inference__traced_save_1559833
file_prefix'
#savev2_variable_read_readvariableop	G
Csavev2_qnetwork_encodingnetwork_dense_20_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_20_bias_read_readvariableop7
3savev2_qnetwork_dense_21_kernel_read_readvariableop5
1savev2_qnetwork_dense_21_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_dccbb171041d44bc8f7006cbe71e168a/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slicesŌ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_20_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_20_bias_read_readvariableop3savev2_qnetwork_dense_21_kernel_read_readvariableop1savev2_qnetwork_dense_21_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*9
_input_shapes(
&: : : d:d:d0:0: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

: d: 

_output_shapes
:d:$ 

_output_shapes

:d0: 

_output_shapes
:0:

_output_shapes
: 
®F

)__inference_polymorphic_action_fn_1559669
	time_step
time_step_1
time_step_2
time_step_3
time_step_4D
@qnetwork_encodingnetwork_dense_20_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_20_biasadd_readvariableop_resource4
0qnetwork_dense_21_matmul_readvariableop_resource5
1qnetwork_dense_21_biasadd_readvariableop_resource
identity§
)QNetwork/EncodingNetwork/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2+
)QNetwork/EncodingNetwork/flatten_10/ConstŲ
+QNetwork/EncodingNetwork/flatten_10/ReshapeReshapetime_step_32QNetwork/EncodingNetwork/flatten_10/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2-
+QNetwork/EncodingNetwork/flatten_10/ReshapeĻ
&QNetwork/EncodingNetwork/dense_20/CastCast4QNetwork/EncodingNetwork/flatten_10/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’ 2(
&QNetwork/EncodingNetwork/dense_20/Castó
7QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_20_matmul_readvariableop_resource*
_output_shapes

: d*
dtype029
7QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOpż
(QNetwork/EncodingNetwork/dense_20/MatMulMatMul*QNetwork/EncodingNetwork/dense_20/Cast:y:0?QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’d2*
(QNetwork/EncodingNetwork/dense_20/MatMulņ
8QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02:
8QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOp
)QNetwork/EncodingNetwork/dense_20/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_20/MatMul:product:0@QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’d2+
)QNetwork/EncodingNetwork/dense_20/BiasAdd¾
&QNetwork/EncodingNetwork/dense_20/ReluRelu2QNetwork/EncodingNetwork/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’d2(
&QNetwork/EncodingNetwork/dense_20/ReluĆ
'QNetwork/dense_21/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_21_matmul_readvariableop_resource*
_output_shapes

:d0*
dtype02)
'QNetwork/dense_21/MatMul/ReadVariableOp×
QNetwork/dense_21/MatMulMatMul4QNetwork/EncodingNetwork/dense_20/Relu:activations:0/QNetwork/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’02
QNetwork/dense_21/MatMulĀ
(QNetwork/dense_21/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_21_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(QNetwork/dense_21/BiasAdd/ReadVariableOpÉ
QNetwork/dense_21/BiasAddBiasAdd"QNetwork/dense_21/MatMul:product:00QNetwork/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’02
QNetwork/dense_21/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ’2
Constb
CastCasttime_step_4*

DstT0
*

SrcT0*'
_output_shapes
:’’’’’’’’’02
Cast
SelectV2SelectV2Cast:y:0"QNetwork/dense_21/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’02

SelectV2
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2%
#Categorical_1/mode/ArgMax/dimensionÆ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:’’’’’’’’’2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/x“
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2É
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsĻ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisŖ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatĪ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:’’’’’’’’’2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3¢
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack¦
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1¦
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ź
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1Š
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:’’’’’’’’’2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :/2
clip_by_value/Minimum/y²
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’0:::::N J
#
_output_shapes
:’’’’’’’’’
#
_user_specified_name	time_step:NJ
#
_output_shapes
:’’’’’’’’’
#
_user_specified_name	time_step:NJ
#
_output_shapes
:’’’’’’’’’
#
_user_specified_name	time_step:ZV
/
_output_shapes
:’’’’’’’’’
#
_user_specified_name	time_step:RN
'
_output_shapes
:’’’’’’’’’0
#
_user_specified_name	time_step
ö
Æ
#__inference__traced_restore_1559858
file_prefix
assignvariableop_variable?
;assignvariableop_1_qnetwork_encodingnetwork_dense_20_kernel=
9assignvariableop_2_qnetwork_encodingnetwork_dense_20_bias/
+assignvariableop_3_qnetwork_dense_21_kernel-
)assignvariableop_4_qnetwork_dense_21_bias

identity_6¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slicesÉ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ą
AssignVariableOp_1AssignVariableOp;assignvariableop_1_qnetwork_encodingnetwork_dense_20_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¾
AssignVariableOp_2AssignVariableOp9assignvariableop_2_qnetwork_encodingnetwork_dense_20_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3°
AssignVariableOp_3AssignVariableOp+assignvariableop_3_qnetwork_dense_21_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4®
AssignVariableOp_4AssignVariableOp)assignvariableop_4_qnetwork_dense_21_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpĻ

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5Į

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ś
e
+__inference_function_with_signature_1559718
unknown
identity	¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *%
f R
__inference_<lambda>_10864812
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall

7
%__inference_get_initial_state_1559705

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Ē
'
%__inference_signature_wrapper_1559733
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_15597292
PartitionedCall*
_input_shapes 
®
7
%__inference_signature_wrapper_1559711

batch_size
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_15597062
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size

_
%__inference_signature_wrapper_1559726
unknown
identity	¢StatefulPartitionedCall“
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_15597182
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
¾
-
+__inference_function_with_signature_1559729ł
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *%
f R
__inference_<lambda>_10864842
PartitionedCall*
_input_shapes 
Ó
L
__inference_<lambda>_1086481
readvariableop_resource
identity	p
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpY
IdentityIdentityReadVariableOp:value:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
Ī(

/__inference_polymorphic_distribution_fn_1086578
	step_type

reward
discount
observation_0
observation_1D
@qnetwork_encodingnetwork_dense_20_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_20_biasadd_readvariableop_resource4
0qnetwork_dense_21_matmul_readvariableop_resource5
1qnetwork_dense_21_biasadd_readvariableop_resource
identity§
)QNetwork/EncodingNetwork/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2+
)QNetwork/EncodingNetwork/flatten_10/ConstŚ
+QNetwork/EncodingNetwork/flatten_10/ReshapeReshapeobservation_02QNetwork/EncodingNetwork/flatten_10/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2-
+QNetwork/EncodingNetwork/flatten_10/ReshapeĻ
&QNetwork/EncodingNetwork/dense_20/CastCast4QNetwork/EncodingNetwork/flatten_10/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’ 2(
&QNetwork/EncodingNetwork/dense_20/Castó
7QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_20_matmul_readvariableop_resource*
_output_shapes

: d*
dtype029
7QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOpż
(QNetwork/EncodingNetwork/dense_20/MatMulMatMul*QNetwork/EncodingNetwork/dense_20/Cast:y:0?QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’d2*
(QNetwork/EncodingNetwork/dense_20/MatMulņ
8QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02:
8QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOp
)QNetwork/EncodingNetwork/dense_20/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_20/MatMul:product:0@QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’d2+
)QNetwork/EncodingNetwork/dense_20/BiasAdd¾
&QNetwork/EncodingNetwork/dense_20/ReluRelu2QNetwork/EncodingNetwork/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’d2(
&QNetwork/EncodingNetwork/dense_20/ReluĆ
'QNetwork/dense_21/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_21_matmul_readvariableop_resource*
_output_shapes

:d0*
dtype02)
'QNetwork/dense_21/MatMul/ReadVariableOp×
QNetwork/dense_21/MatMulMatMul4QNetwork/EncodingNetwork/dense_20/Relu:activations:0/QNetwork/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’02
QNetwork/dense_21/MatMulĀ
(QNetwork/dense_21/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_21_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(QNetwork/dense_21/BiasAdd/ReadVariableOpÉ
QNetwork/dense_21/BiasAddBiasAdd"QNetwork/dense_21/MatMul:product:00QNetwork/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’02
QNetwork/dense_21/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ’2
Constd
CastCastobservation_1*

DstT0
*

SrcT0*'
_output_shapes
:’’’’’’’’’02
Cast
SelectV2SelectV2Cast:y:0"QNetwork/dense_21/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’02

SelectV2
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2%
#Categorical_1/mode/ArgMax/dimensionÆ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:’’’’’’’’’2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtoln
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/atoln
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/rtolk
IdentityIdentityCategorical_1/mode/Cast:y:0*
T0*#
_output_shapes
:’’’’’’’’’2

Identityn
Deterministic_2/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/atoln
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/rtol"
identityIdentity:output:0*~
_input_shapesm
k:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’0:::::N J
#
_output_shapes
:’’’’’’’’’
#
_user_specified_name	step_type:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_namereward:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
discount:^Z
/
_output_shapes
:’’’’’’’’’
'
_user_specified_nameobservation/0:VR
'
_output_shapes
:’’’’’’’’’0
'
_user_specified_nameobservation/1

į
+__inference_function_with_signature_1559680
	step_type

reward
discount
observation_0
observation_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_0observation_1unknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *2
f-R+
)__inference_polymorphic_action_fn_15596692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’0::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:’’’’’’’’’
%
_user_specified_name0/step_type:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
0/reward:OK
#
_output_shapes
:’’’’’’’’’
$
_user_specified_name
0/discount:`\
/
_output_shapes
:’’’’’’’’’
)
_user_specified_name0/observation/0:XT
'
_output_shapes
:’’’’’’’’’0
)
_user_specified_name0/observation/1
3

__inference_<lambda>_1086484*
_input_shapes 

7
%__inference_get_initial_state_1086475

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size

Ū
%__inference_signature_wrapper_1559699
discount
observation_0
observation_1

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_0observation_1unknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *4
f/R-
+__inference_function_with_signature_15596802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’0:’’’’’’’’’:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:’’’’’’’’’
$
_user_specified_name
0/discount:`\
/
_output_shapes
:’’’’’’’’’
)
_user_specified_name0/observation/0:XT
'
_output_shapes
:’’’’’’’’’0
)
_user_specified_name0/observation/1:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
0/reward:PL
#
_output_shapes
:’’’’’’’’’
%
_user_specified_name0/step_type
®
=
+__inference_function_with_signature_1559706

batch_size
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_get_initial_state_15597052
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ŖG
·
)__inference_polymorphic_action_fn_1559789
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_0
time_step_observation_1D
@qnetwork_encodingnetwork_dense_20_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_20_biasadd_readvariableop_resource4
0qnetwork_dense_21_matmul_readvariableop_resource5
1qnetwork_dense_21_biasadd_readvariableop_resource
identity§
)QNetwork/EncodingNetwork/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2+
)QNetwork/EncodingNetwork/flatten_10/Constä
+QNetwork/EncodingNetwork/flatten_10/ReshapeReshapetime_step_observation_02QNetwork/EncodingNetwork/flatten_10/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2-
+QNetwork/EncodingNetwork/flatten_10/ReshapeĻ
&QNetwork/EncodingNetwork/dense_20/CastCast4QNetwork/EncodingNetwork/flatten_10/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’ 2(
&QNetwork/EncodingNetwork/dense_20/Castó
7QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_20_matmul_readvariableop_resource*
_output_shapes

: d*
dtype029
7QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOpż
(QNetwork/EncodingNetwork/dense_20/MatMulMatMul*QNetwork/EncodingNetwork/dense_20/Cast:y:0?QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’d2*
(QNetwork/EncodingNetwork/dense_20/MatMulņ
8QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02:
8QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOp
)QNetwork/EncodingNetwork/dense_20/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_20/MatMul:product:0@QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’d2+
)QNetwork/EncodingNetwork/dense_20/BiasAdd¾
&QNetwork/EncodingNetwork/dense_20/ReluRelu2QNetwork/EncodingNetwork/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’d2(
&QNetwork/EncodingNetwork/dense_20/ReluĆ
'QNetwork/dense_21/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_21_matmul_readvariableop_resource*
_output_shapes

:d0*
dtype02)
'QNetwork/dense_21/MatMul/ReadVariableOp×
QNetwork/dense_21/MatMulMatMul4QNetwork/EncodingNetwork/dense_20/Relu:activations:0/QNetwork/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’02
QNetwork/dense_21/MatMulĀ
(QNetwork/dense_21/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_21_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(QNetwork/dense_21/BiasAdd/ReadVariableOpÉ
QNetwork/dense_21/BiasAddBiasAdd"QNetwork/dense_21/MatMul:product:00QNetwork/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’02
QNetwork/dense_21/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ’2
Constn
CastCasttime_step_observation_1*

DstT0
*

SrcT0*'
_output_shapes
:’’’’’’’’’02
Cast
SelectV2SelectV2Cast:y:0"QNetwork/dense_21/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’02

SelectV2
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2%
#Categorical_1/mode/ArgMax/dimensionÆ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:’’’’’’’’’2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/x“
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2É
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsĻ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisŖ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatĪ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:’’’’’’’’’2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3¢
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack¦
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1¦
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ź
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1Š
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:’’’’’’’’’2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :/2
clip_by_value/Minimum/y²
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’0:::::X T
#
_output_shapes
:’’’’’’’’’
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:’’’’’’’’’
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:’’’’’’’’’
,
_user_specified_nametime_step/discount:hd
/
_output_shapes
:’’’’’’’’’
1
_user_specified_nametime_step/observation/0:`\
'
_output_shapes
:’’’’’’’’’0
1
_user_specified_nametime_step/observation/1
²F

)__inference_polymorphic_action_fn_1086541
	step_type

reward
discount
observation_0
observation_1D
@qnetwork_encodingnetwork_dense_20_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_20_biasadd_readvariableop_resource4
0qnetwork_dense_21_matmul_readvariableop_resource5
1qnetwork_dense_21_biasadd_readvariableop_resource
identity§
)QNetwork/EncodingNetwork/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2+
)QNetwork/EncodingNetwork/flatten_10/ConstŚ
+QNetwork/EncodingNetwork/flatten_10/ReshapeReshapeobservation_02QNetwork/EncodingNetwork/flatten_10/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2-
+QNetwork/EncodingNetwork/flatten_10/ReshapeĻ
&QNetwork/EncodingNetwork/dense_20/CastCast4QNetwork/EncodingNetwork/flatten_10/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’ 2(
&QNetwork/EncodingNetwork/dense_20/Castó
7QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_20_matmul_readvariableop_resource*
_output_shapes

: d*
dtype029
7QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOpż
(QNetwork/EncodingNetwork/dense_20/MatMulMatMul*QNetwork/EncodingNetwork/dense_20/Cast:y:0?QNetwork/EncodingNetwork/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’d2*
(QNetwork/EncodingNetwork/dense_20/MatMulņ
8QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02:
8QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOp
)QNetwork/EncodingNetwork/dense_20/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_20/MatMul:product:0@QNetwork/EncodingNetwork/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’d2+
)QNetwork/EncodingNetwork/dense_20/BiasAdd¾
&QNetwork/EncodingNetwork/dense_20/ReluRelu2QNetwork/EncodingNetwork/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’d2(
&QNetwork/EncodingNetwork/dense_20/ReluĆ
'QNetwork/dense_21/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_21_matmul_readvariableop_resource*
_output_shapes

:d0*
dtype02)
'QNetwork/dense_21/MatMul/ReadVariableOp×
QNetwork/dense_21/MatMulMatMul4QNetwork/EncodingNetwork/dense_20/Relu:activations:0/QNetwork/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’02
QNetwork/dense_21/MatMulĀ
(QNetwork/dense_21/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_21_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(QNetwork/dense_21/BiasAdd/ReadVariableOpÉ
QNetwork/dense_21/BiasAddBiasAdd"QNetwork/dense_21/MatMul:product:00QNetwork/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’02
QNetwork/dense_21/BiasAddS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ’2
Constd
CastCastobservation_1*

DstT0
*

SrcT0*'
_output_shapes
:’’’’’’’’’02
Cast
SelectV2SelectV2Cast:y:0"QNetwork/dense_21/BiasAdd:output:0Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’02

SelectV2
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2%
#Categorical_1/mode/ArgMax/dimensionÆ
Categorical_1/mode/ArgMaxArgMaxSelectV2:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:’’’’’’’’’2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/x“
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2É
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsĻ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisŖ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatĪ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:’’’’’’’’’2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3¢
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack¦
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1¦
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ź
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1Š
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:’’’’’’’’’2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :/2
clip_by_value/Minimum/y²
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’0:::::N J
#
_output_shapes
:’’’’’’’’’
#
_user_specified_name	step_type:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_namereward:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
discount:^Z
/
_output_shapes
:’’’’’’’’’
'
_user_specified_nameobservation/0:VR
'
_output_shapes
:’’’’’’’’’0
'
_user_specified_nameobservation/1"øL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
action
4

0/discount&
action_0/discount:0’’’’’’’’’
J
0/observation/07
action_0/observation/0:0’’’’’’’’’
B
0/observation/1/
action_0/observation/1:0’’’’’’’’’0
0
0/reward$
action_0/reward:0’’’’’’’’’
6
0/step_type'
action_0/step_type:0’’’’’’’’’6
action,
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:øe
Ķ

train_step
metadata
model_variables
_all_assets

signatures

?action
@distribution
Aget_initial_state
Bget_metadata
Cget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
=
0
1
2
	3"
trackable_tuple_wrapper
'

0"
trackable_list_wrapper
`

Daction
Eget_initial_state
Fget_train_step
Gget_metadata"
signature_map
::8 d2(QNetwork/EncodingNetwork/dense_20/kernel
4:2d2&QNetwork/EncodingNetwork/dense_20/bias
*:(d02QNetwork/dense_21/kernel
$:"02QNetwork/dense_21/bias
1
ref
1"
trackable_tuple_wrapper
.

_q_network"
_generic_user_object
Ć
_encoder
_q_value_layer
regularization_losses
trainable_variables
	variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"
_tf_keras_layerų{"class_name": "QNetwork", "name": "QNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ė
_postprocessing_layers
regularization_losses
trainable_variables
	variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__" 
_tf_keras_layer{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ģ

kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"§
_tf_keras_layer{"class_name": "Dense", "name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 48, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 100]}}
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
­
layer_metrics
metrics

layers
layer_regularization_losses
 non_trainable_variables
regularization_losses
trainable_variables
	variables
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
#layer_metrics
$metrics

%layers
&layer_regularization_losses
'non_trainable_variables
regularization_losses
trainable_variables
	variables
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
­
(layer_metrics
)metrics

*layers
+layer_regularization_losses
,non_trainable_variables
regularization_losses
trainable_variables
	variables
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
č
-regularization_losses
.trainable_variables
/	variables
0	keras_api
*N&call_and_return_all_conditional_losses
O__call__"Ł
_tf_keras_layeræ{"class_name": "Flatten", "name": "flatten_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_10", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ź

kernel
bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
*P&call_and_return_all_conditional_losses
Q__call__"„
_tf_keras_layer{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
5layer_metrics
6metrics

7layers
8layer_regularization_losses
9non_trainable_variables
-regularization_losses
.trainable_variables
/	variables
O__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
:layer_metrics
;metrics

<layers
=layer_regularization_losses
>non_trainable_variables
1regularization_losses
2trainable_variables
3	variables
Q__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
)__inference_polymorphic_action_fn_1559789
)__inference_polymorphic_action_fn_1086541±
Ŗ²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
č2å
/__inference_polymorphic_distribution_fn_1086578±
Ŗ²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
%__inference_get_initial_state_1086475¦
²
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 B
__inference_<lambda>_1086484
 B
__inference_<lambda>_1086481
nBl
%__inference_signature_wrapper_1559699
0/discount0/observation/00/observation/10/reward0/step_type
7B5
%__inference_signature_wrapper_1559711
batch_size
)B'
%__inference_signature_wrapper_1559726
)B'
%__inference_signature_wrapper_1559733
ę2ćą
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ę2ćą
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ę2ćą
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ę2ćą
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ;
__inference_<lambda>_1086481¢

¢ 
Ŗ " 	4
__inference_<lambda>_1086484¢

¢ 
Ŗ "Ŗ R
%__inference_get_initial_state_1086475)"¢
¢


batch_size 
Ŗ "¢ ”
)__inference_polymorphic_action_fn_1086541ó	¢
¢
ž²ś
TimeStep,
	step_type
	step_type’’’’’’’’’&
reward
reward’’’’’’’’’*
discount
discount’’’’’’’’’l
observation]¢Z
/,
observation/0’’’’’’’’’
'$
observation/1’’’’’’’’’0
¢ 
Ŗ "R²O

PolicyStep&
action
action’’’’’’’’’
state¢ 
info¢ Ō
)__inference_polymorphic_action_fn_1559789¦	É¢Å
½¢¹
±²­
TimeStep6
	step_type)&
time_step/step_type’’’’’’’’’0
reward&#
time_step/reward’’’’’’’’’4
discount(%
time_step/discount’’’’’’’’’
observationq¢n
96
time_step/observation/0’’’’’’’’’
1.
time_step/observation/1’’’’’’’’’0
¢ 
Ŗ "R²O

PolicyStep&
action
action’’’’’’’’’
state¢ 
info¢ 
/__inference_polymorphic_distribution_fn_1086578Ś	¢
¢
ž²ś
TimeStep,
	step_type
	step_type’’’’’’’’’&
reward
reward’’’’’’’’’*
discount
discount’’’’’’’’’l
observation]¢Z
/,
observation/0’’’’’’’’’
'$
observation/1’’’’’’’’’0
¢ 
Ŗ "ø²“

PolicyStep
action’ūšįĆŪ¢×
`
C¢@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*Ŗ'
%
loc
Identity’’’’’’’’’
`Ŗ]

allow_nan_statsp


atol
 

namejDeterministic


rtol
 

validate_argsp _DistributionTypeSpec
state¢ 
info¢ 
%__inference_signature_wrapper_1559699Ų	¢¢
¢ 
Ŗ
.

0/discount 

0/discount’’’’’’’’’
D
0/observation/01.
0/observation/0’’’’’’’’’
<
0/observation/1)&
0/observation/1’’’’’’’’’0
*
0/reward
0/reward’’’’’’’’’
0
0/step_type!
0/step_type’’’’’’’’’"+Ŗ(
&
action
action’’’’’’’’’`
%__inference_signature_wrapper_155971170¢-
¢ 
&Ŗ#
!

batch_size

batch_size "Ŗ Y
%__inference_signature_wrapper_15597260¢

¢ 
Ŗ "Ŗ

int64
int64 	=
%__inference_signature_wrapper_1559733¢

¢ 
Ŗ "Ŗ 