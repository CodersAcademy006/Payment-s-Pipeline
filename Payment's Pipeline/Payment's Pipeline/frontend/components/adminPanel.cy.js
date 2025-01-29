describe('Admin Panel Tests', () => {
    beforeEach(() => {
      cy.loginAsAdmin(); // Implement custom command for admin login
      cy.visit('/admin-panel');
    });
  
    it('should load transactions', () => {
      cy.get('[data-cy="transaction-row"]').should('have.length.at.least', 1);
    });
  
    it('should filter transactions', () => {
      cy.get('[data-cy="search-input"]').type('test');
      cy.get('[data-cy="transaction-row"]').should('exist');
    });
  });


// Add to AdminPanel component
import { useTranslation } from 'react-i18next';

// Inside component:
const { t } = useTranslation();

// Replace hardcoded text with translations:
<Typography variant="h4" component="h1" gutterBottom>
  {t('adminPanel.title')}
</Typography>