use AI::MXNet 'mx';
use AI::MXNet::TestUtils qw(check_numeric_gradient); ## not implemented yet, will croak
use Test::More 'no_plan';

# adapted from tests/python/unittest/operator.py
# see python/mxnet/operator.py
# see python/mxnet/ndarray/ndarray.py for the gluon part
# see https://github.com/apache/incubator-mxnet/pull/7828 for some context of how it's done on C++ side

package AI::MXNet::CustomOp::Sqr;
use AI::MXNet::Function::Parameters;
use Mouse;
extends 'AI::MXNet::CustomOp';
method forward($is_train, $req, $in_data, $out_data, $aux)
{
    $self->assign($out_data->[0], $req->[0], $in_data->[0]*$in_data->[0]);
}

method backward($req, $out_grad, $in_data, $out_data, $in_grad, $aux)
{
    $self->assign($in_grad->[0], $req->[0], 2*$in_data->[0]*$out_grad->[0]);
}

__PACKAGE__->register;

package AI::MXNet::CustomOpProp::SqrProp;
use Mouse;
extends 'AI::MXNet::CustomOpProp';
has '+needs_top_grad' => (default => 1);

method list_arguments() { ['data'] }

method list_outputs()   { ['output'] }

method infer_shape($in_shape)
{
    return ($in_shape, [$in_shape->[0]], []);
}

method infer_type($in_type)
{
    return ($in_type, [$in_type->[0]], []);
}

method create_operator(AI::NXNet::Context $ctx, ArrayRef[Shape] $shapes, ArrayRef[Dtype] $dtypes)
{
    return AI::MXNet::CustomOp::Sqr->new;
}

__PACKAGE__->register;

sub test_custom_op
{
    my $data = mx->symbol->Variable('data');
    my $op = mx->symbol->Custom(data=>$data, name=>'sqr', op_type=>'sqr');
    my $x = mx->nd->uniform(-1, 1, { shape=>[4, 10] });
    check_numeric_gradient($op, [$x]);

    $data = mx->symbol->Variable('data');
    $data = mx->symbol->cast($data, dtype=>'float64');
    $op = mx->symbol->Custom(data=>$data, name=>'sqr', op_type=>'sqr');
    $op = mx->symbol->cast($op, dtype=>'float32');
    $x = mx->nd->uniform(-1, 1, { shape=>[4, 10] });
    check_numeric_gradient($op, [$x]);

    ## related to gluon branch 
    my $dx = mx->nd->zeros_like($x);
    mx->autograd->mark_variables([$x], [$dx]);
    mx->autograd->train_section(sub {
        my $y = mx->nd->Custom($x, op_type=>'sqr');
        $y->backward
    });
}

test_custom_op();
