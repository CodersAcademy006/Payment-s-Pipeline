import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import AdminPanel from './admin_panel';
import axios from 'axios';
import { Provider } from 'react-redux';
import store from '../redux/store';
import { BrowserRouter } from 'react-router-dom';

jest.mock('axios');

const mockTransactions = [
  {
    id: '1',
    shortId: 'TX123',
    amount: 100,
    currency: 'USD',
    status: 'completed',
    paymentMethod: 'Credit Card',
    createdAt: '2023-01-01T00:00:00Z'
  }
];

describe('AdminPanel', () => {
  beforeEach(() => {
    axios.get.mockResolvedValue({
      data: { transactions: mockTransactions, totalCount: 1 }
    });
  });

  test('renders and fetches transactions', async () => {
    render(
      <BrowserRouter>
        <Provider store={store}>
          <AdminPanel />
        </Provider>
      </BrowserRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('TX123')).toBeInTheDocument();
    });
  });

  test('handles search filter', async () => {
    render(<AdminPanel />);
    const searchInput = screen.getByLabelText('Search Transactions');
    
    fireEvent.change(searchInput, { target: { value: 'test' } });
    await waitFor(() => {
      expect(axios.get).toHaveBeenCalledWith(
        expect.stringContaining('/api/admin/transactions'),
        expect.objectContaining({
          params: expect.objectContaining({ search: 'test' })
        })
      );
    });
  });
});