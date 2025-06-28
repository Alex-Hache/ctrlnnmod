import torch
torch.set_default_dtype(torch.float64)  # Higher precision needed for initialization
from ctrlnmod.models.ssmodels.continuous.general import GNSSM
from ctrlnmod.models.feedforward.lbdn import LBDN, FFNN
from ctrlnmod.models.ssmodels.continuous.linear import SSLinear, L2BoundedLinear, H2Linear
import pytest
import numpy as np


@pytest.fixture
def dims():
    return {
        'nu': 2,
        'ny': 2,
        'nx': 4,
        'hidden_layers': [32, 16]
    }


@pytest.fixture
def lin_model_types():
    return [None, 'h2', 'l2']


@pytest.fixture
def lipschitz_constants():
    return [
        {'lip_fx': 2.0, 'lip_fu': 1.0, 'lip_hx': None},
        {'lip_fx': 2.0, 'lip_fu': None, 'lip_hx': None},
        {'lip_fx': None, 'lip_fu': 1.0, 'lip_hx': None},
        {'lip_fx': None, 'lip_fu': None, 'lip_hx': 2.0},
        {'lip_fx': 1.0, 'lip_fu': 3.0, 'lip_hx': 2.0},
        None  # No Lipschitz constraints
    ]


@pytest.fixture
def valid_matrices(dims):
    """Generate stable matrices for initialization tests"""
    nx = dims['nx']
    nu = dims['nu']
    ny = dims['ny']
    
    # Create a stable A matrix
    A = torch.randn(nx, nx) * 0.1
    A = A - torch.eye(nx) * 0.5  # Make it stable
    B = torch.randn(nx, nu)
    C = torch.randn(ny, nx)
    
    return A, B, C


class TestGNSSMConstructor:
    """Test all constructor variations and exception handling"""
    
    def test_basic_constructor(self, dims):
        """Test basic constructor with minimal parameters"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        model = GNSSM(nu, ny, nx)
        
        assert model.nu == nu
        assert model.ny == ny
        assert model.nx == nx
        assert model.hidden_layers == [16]  # default
        assert model.act_f_str == 'relu'  # default
        assert model.out_eq_nl == False  # default
        assert model.bias == True  # default
        assert model.lin_model_type is None
        assert isinstance(model.linmod, SSLinear)
        assert isinstance(model.fxu, FFNN)
        assert not hasattr(model, 'hx')
    
    def test_constructor_with_all_params(self, dims):
        """Test constructor with all parameters specified"""
        nu, ny, nx, hidden_layers = dims['nu'], dims['ny'], dims['nx'], dims['hidden_layers']
        
        model = GNSSM(
            nu=nu, ny=ny, nx=nx,
            hidden_layers=hidden_layers,
            act_f='tanh',
            out_eq_nl=True,
            alpha=0.1,
            bias=False,
            lin_model_type='l2',
            lip={'lip_fx': 1.0, 'lip_fu': 2.0, 'lip_hx': 1.5},
            param='sqrtm',
            gamma=3.0
        )
        
        assert model.hidden_layers == hidden_layers
        assert model.act_f_str == 'tanh'
        assert model.out_eq_nl == True
        assert model.bias == False
        assert model.alpha == 0.1
        assert model.gamma == 3.0
        assert isinstance(model.linmod, L2BoundedLinear)
        assert isinstance(model.fxu, LBDN)
        assert hasattr(model, 'hx')
        assert isinstance(model.hx, LBDN)
    
    def test_constructor_lin_model_types(self, dims, lin_model_types):
        """Test all linear model types"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        
        for lin_type in lin_model_types:
            if lin_type is None:
                model = GNSSM(nu, ny, nx, lin_model_type=lin_type)
                assert isinstance(model.linmod, SSLinear)
            elif lin_type == 'h2':
                model = GNSSM(nu, ny, nx, lin_model_type=lin_type, gamma2=2.0)
                assert isinstance(model.linmod, H2Linear)
                assert model.gamma2 == 2.0
            elif lin_type == 'l2':
                model = GNSSM(nu, ny, nx, lin_model_type=lin_type, gamma=3.0)
                assert isinstance(model.linmod, L2BoundedLinear)
                assert model.gamma == 3.0
    
    def test_constructor_lipschitz_variations(self, dims, lipschitz_constants):
        """Test all Lipschitz constant variations"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        
        for lip_dict in lipschitz_constants:
            if lip_dict is None:
                # No Lipschitz constraints
                model = GNSSM(nu, ny, nx, lip=lip_dict)
                assert isinstance(model.fxu, FFNN)
            elif lip_dict['lip_fx'] is None and lip_dict['lip_fu'] is None:
                needs_hx = lip_dict.get('lip_hx') is not None
                if needs_hx:
                    model = GNSSM(nu, ny, nx, lip=lip_dict, out_eq_nl=True)
                    assert isinstance(model.fxu, FFNN)
                    assert isinstance(model.hx, LBDN)
                else:
                    model = GNSSM(nu, ny, nx, lip=lip_dict)
                    assert isinstance(model.fxu, FFNN)
            else:
                needs_hx = lip_dict.get('lip_hx') is not None
                if needs_hx:
                    model = GNSSM(nu, ny, nx, lip=lip_dict, out_eq_nl=True)
                    assert isinstance(model.hx, LBDN)
                else:
                    model = GNSSM(nu, ny, nx, lip=lip_dict, out_eq_nl=False)
                    
                assert isinstance(model.fxu, LBDN)
                
                # Check scale values
                scales = model.fxu.scale.tolist()
                lip_fx = lip_dict.get('lip_fx')
                lip_fu = lip_dict.get('lip_fu')
                
                # Handle default values as per constructor logic
                if lip_fx is None and lip_fu is not None:
                    lip_fx = lip_fu
                elif lip_fu is None and lip_fx is not None:
                    lip_fu = lip_fx
                
                if lip_fx is not None:
                    assert all(s == lip_fx for s in scales[:nx])
                if lip_fu is not None:
                    assert all(s == lip_fu for s in scales[nx:])
    
    def test_constructor_exceptions(self, dims):
        """Test exception handling in constructor"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        
        # Test H2 without gamma2
        with pytest.raises(ValueError, match="Parameter 'gamma2' must be provided"):
            GNSSM(nu, ny, nx, lin_model_type='h2')
        
        # Test L2 without gamma
        with pytest.raises(ValueError, match="Parameter 'gamma' must be provided"):
            GNSSM(nu, ny, nx, lin_model_type='l2')
        
        # Test invalid parameterization
        with pytest.raises(NotImplementedError, match="Parameterization"):
            GNSSM(nu, ny, nx, lin_model_type='invalid')
        
        # Test invalid Lipschitz dictionary (all None)
        with pytest.raises(ValueError, match="Invalid dictionnary found"):
            GNSSM(nu, ny, nx, lip={'lip_fx': None, 'lip_fu': None, 'lip_hx': None})
        
        # Test output nonlinearity with Lipschitz but missing lip_hx
        with pytest.raises(ValueError, match="Please specify a value for output equation"):
            GNSSM(nu, ny, nx, out_eq_nl=True, lip={'lip_fx': 1.0, 'lip_fu': 1.0, 'lip_hx': None})


class TestGNSSMInitialization:
    """Test model initialization methods"""
    
    def test_init_weights_ss_linear(self, dims, valid_matrices):
        """Test weight initialization for SSLinear models"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        A0, B0, C0 = valid_matrices
        
        model = GNSSM(nu, ny, nx, lin_model_type=None)
        model.init_weights_(A0, B0, C0)
        
        # Check that parameters are initialized (not zeros)
        linear_params = list(model.linmod.parameters())
        assert len(linear_params) > 0
        
        # Check nonlinear part initialization
        fxu_params = list(model.fxu.parameters())
        assert len(fxu_params) > 0
        assert not torch.allclose(fxu_params[0], torch.zeros_like(fxu_params[0]))
    
    def test_init_weights_l2_bounded(self, dims, valid_matrices):
        """Test weight initialization for L2BoundedLinear models"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        A0, B0, C0 = valid_matrices
        
        model = GNSSM(nu, ny, nx, lin_model_type='l2', gamma=5.0)
        model.init_weights_(A0, B0, C0)
        
        # Verify initialization completed without errors
        assert model.linmod is not None
        assert model.fxu is not None
    
    def test_init_weights_h2(self, dims, valid_matrices):
        """Test weight initialization for H2Linear models"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        A0, B0, C0 = valid_matrices
        
        model = GNSSM(nu, ny, nx, lin_model_type='h2', gamma2=2.0)
        model.init_weights_(A0, B0, C0)
        
        # Verify initialization completed without errors
        assert model.linmod is not None
        assert model.fxu is not None
    
    def test_init_weights_with_output_nl(self, dims, valid_matrices):
        """Test initialization with output nonlinearity"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        A0, B0, C0 = valid_matrices
        
        model = GNSSM(nu, ny, nx, out_eq_nl=True)
        model.init_weights_(A0, B0, C0)
        
        # Check that output nonlinearity is initialized
        assert hasattr(model, 'hx')
        hx_params = list(model.hx.parameters())
        assert len(hx_params) > 0
        assert not torch.allclose(hx_params[0], torch.zeros_like(hx_params[0]))


class TestGNSSMForward:
    """Test forward pass functionality"""
    
    def test_forward_basic(self, dims):
        """Test basic forward pass"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        batch_size = 3
        
        model = GNSSM(nu, ny, nx)
        
        u = torch.randn(batch_size, nu)
        x = torch.randn(batch_size, nx)
        
        dx, y = model.forward(u, x)
        
        assert dx.shape == (batch_size, nx)
        assert y.shape == (batch_size, ny)
    
    def test_forward_with_output_nl(self, dims):
        """Test forward pass with output nonlinearity"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        batch_size = 3
        
        model = GNSSM(nu, ny, nx, out_eq_nl=True)
        
        u = torch.randn(batch_size, nu)
        x = torch.randn(batch_size, nx)
        
        dx, y = model.forward(u, x)
        
        assert dx.shape == (batch_size, nx)
        assert y.shape == (batch_size, ny)
    
    def test_forward_with_lipschitz(self, dims):
        """Test forward pass with Lipschitz constraints"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        batch_size = 3
        
        model = GNSSM(nu, ny, nx, 
                     lip={'lip_fx': 1.0, 'lip_fu': 2.0, 'lip_hx': 1.5}, 
                     out_eq_nl=True)
        
        u = torch.randn(batch_size, nu)
        x = torch.randn(batch_size, nx)
        
        dx, y = model.forward(u, x)
        
        assert dx.shape == (batch_size, nx)
        assert y.shape == (batch_size, ny)
    
    def test_forward_different_batch_sizes(self, dims):
        """Test forward pass with different batch sizes"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        model = GNSSM(nu, ny, nx)
        
        for batch_size in [1, 5, 10]:
            u = torch.randn(batch_size, nu)
            x = torch.randn(batch_size, nx)
            
            dx, y = model.forward(u, x)
            
            assert dx.shape == (batch_size, nx)
            assert y.shape == (batch_size, ny)


class TestGNSSMUtilities:
    """Test utility methods"""
    
    def test_repr_and_str(self, dims):
        """Test string representations"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        
        # Basic model
        model = GNSSM(nu, ny, nx)
        repr_str = repr(model)
        str_str = str(model)
        
        assert f"nu={nu}" in repr_str
        assert f"ny={ny}" in repr_str
        assert f"nx={nx}" in repr_str
        assert repr_str == str_str
        
        # Model with Lipschitz constraints
        model_lip = GNSSM(nu, ny, nx, 
                         lip={'lip_fx': 1.0, 'lip_fu': 2.0, 'lip_hx': 1.5}, 
                         out_eq_nl=True)
        repr_lip = repr(model_lip)
        
        assert "lip_fx=1.0" in repr_lip
        assert "lip_fu=2.0" in repr_lip
        assert "lip_hx=1.5" in repr_lip
    
    def test_clone(self, dims):
        """Test model cloning"""
        nu, ny, nx, hidden_layers = dims['nu'], dims['ny'], dims['nx'], dims['hidden_layers']
        
        # Create original model with various parameters
        original = GNSSM(
            nu=nu, ny=ny, nx=nx,
            hidden_layers=hidden_layers,
            act_f='tanh',
            out_eq_nl=True,
            alpha=0.1,
            lin_model_type='l2',
            lip={'lip_fx': 1.0, 'lip_fu': 2.0, 'lip_hx': 1.5},
            gamma=3.0
        )
        
        # Clone the model
        cloned = original.clone()
        
        # Check that all attributes are preserved
        assert cloned.nu == original.nu
        assert cloned.ny == original.ny
        assert cloned.nx == original.nx
        assert cloned.hidden_layers == original.hidden_layers
        assert cloned.act_f_str == original.act_f_str
        assert cloned.out_eq_nl == original.out_eq_nl
        assert cloned.alpha == original.alpha
        assert cloned.lin_model_type == original.lin_model_type
        assert cloned.gamma == original.gamma
        
        # Check that parameters are copied
        original_params = dict(original.named_parameters())
        cloned_params = dict(cloned.named_parameters())
        
        assert len(original_params) == len(cloned_params)
        for name in original_params:
            assert torch.allclose(original_params[name], cloned_params[name])
        
        # Check that they are independent (modifying one doesn't affect the other)
        with torch.no_grad():
            first_param_name = next(iter(original_params.keys()))
            original_params[first_param_name] += 1.0
        
        assert not torch.allclose(
            original_params[first_param_name], 
            cloned_params[first_param_name]
        )


class TestGNSSMGradients:
    """Test gradient computation"""
    
    def test_gradients_basic(self, dims):
        """Test that gradients can be computed"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        batch_size = 2
        
        model = GNSSM(nu, ny, nx)
        
        u = torch.randn(batch_size, nu)
        x = torch.randn(batch_size, nx)
        
        dx, y = model.forward(u, x)
        
        # Compute some loss
        loss = torch.sum(dx**2) + torch.sum(y**2)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_gradients_with_lipschitz(self, dims):
        """Test gradients with Lipschitz constraints"""
        nu, ny, nx = dims['nu'], dims['ny'], dims['nx']
        batch_size = 2
        
        model = GNSSM(nu, ny, nx, 
                     lip={'lip_fx': 1.0, 'lip_fu': 2.0, 'lip_hx': 1.5}, 
                     out_eq_nl=True)
        
        u = torch.randn(batch_size, nu)
        x = torch.randn(batch_size, nx)
        
        dx, y = model.forward(u, x)
        
        # Compute some loss
        loss = torch.sum(dx**2) + torch.sum(y**2)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


# Integration tests
class TestGNSSMIntegration:
    """Integration tests combining multiple features"""
    
    def test_full_workflow(self, dims, valid_matrices):
        """Test complete workflow: construction, initialization, forward pass"""
        nu, ny, nx, hidden_layers = dims['nu'], dims['ny'], dims['nx'], dims['hidden_layers']
        A0, B0, C0 = valid_matrices
        batch_size = 4
        
        # Create model with all features
        model = GNSSM(
            nu=nu, ny=ny, nx=nx,
            hidden_layers=hidden_layers,
            act_f='tanh',
            out_eq_nl=True,
            lin_model_type='l2',
            lip={'lip_fx': 1.0, 'lip_fu': 2.0, 'lip_hx': 1.5},
            gamma=3.0,
            alpha=0.1
        )
        
        # Initialize
        model.init_weights_(A0, B0, C0)
        
        # Forward pass
        u = torch.randn(batch_size, nu)
        x = torch.randn(batch_size, nx)
        
        dx, y = model.forward(u, x)
        
        assert dx.shape == (batch_size, nx)
        assert y.shape == (batch_size, ny)
        
        # Compute gradients
        loss = torch.sum(dx**2) + torch.sum(y**2)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
        
        # Test cloning
        cloned = model.clone()
        
        # Test that cloned model gives same results
        dx_clone, y_clone = cloned.forward(u, x)
        
        assert torch.allclose(dx, dx_clone, atol=1e-10)
        assert torch.allclose(y, y_clone, atol=1e-10)


class TestLure:

    def test_lure_ffnn(self, dims, valid_matrices):

        nu, ny, nx, hidden_layers = dims['nu'], dims['ny'], dims['nx'], dims['hidden_layers']
        A0, B0, C0 = valid_matrices
        batch_size = 4
        
        # Create model 
        model = GNSSM(
            nu=nu, ny=ny, nx=nx,
            hidden_layers=hidden_layers,
            act_f='tanh',
            out_eq_nl=True,
            lin_model_type='l2',
            gamma=3.0,
            alpha=0.1
        )
        model.init_weights_(A0, B0, C0, init_type='linear')

        # Setting up weights to specific values to debug and asses the lur'e form is correcly built
        for i, layer in enumerate(model.fxu.layers):
            if hasattr(layer, 'weight'):
                print(layer.weight.shape)
                layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight)*i)
        A, B1, B2, C1, C2, D11, D12, D21, D22 = model.to_lure()

        print(A.shape)
        print(B1.shape)
        print(B2.shape)
        print(C1.shape)
        print(C2.shape)
        print(D11.shape)
        print((D11==torch.zeros_like(D11)).all())


if __name__ == "__main__":
    pytest.main([__file__])


