import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import './bloc/family_details_bloc.dart';
import './bloc/family_details_event.dart';
import './bloc/family_details_state.dart';
import './models/family_details_model.dart';
import '../dashboard_page/models/class_model.dart';

class FamilyDetailsPage extends StatefulWidget {
  final String userId;
  final ClassModel? classModel; // Optional, for teacher view context

  const FamilyDetailsPage({
    super.key,
    required this.userId,
    this.classModel,
  });

  @override
  State<FamilyDetailsPage> createState() => _FamilyDetailsPageState();
}

class _FamilyDetailsPageState extends State<FamilyDetailsPage> {
  final _formKey = GlobalKey<FormState>();

  final _fatherName = TextEditingController();
  final _fatherOccupation = TextEditingController();
  final _fatherQualification = TextEditingController();
  final _fatherMobile = TextEditingController();
  final _fatherEmail = TextEditingController();

  final _motherName = TextEditingController();
  final _motherOccupation = TextEditingController();
  final _motherQualification = TextEditingController();
  final _motherMobile = TextEditingController();
  final _motherEmail = TextEditingController();

  @override
  void dispose() {
    _fatherName.dispose();
    _fatherOccupation.dispose();
    _fatherQualification.dispose();
    _fatherMobile.dispose();
    _fatherEmail.dispose();
    _motherName.dispose();
    _motherOccupation.dispose();
    _motherQualification.dispose();
    _motherMobile.dispose();
    _motherEmail.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Family Details')),
      body: BlocListener<FamilyDetailsBloc, FamilyDetailsState>(
        listener: (context, state) {
          if (state is FamilyDetailsSuccess) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text("Family details updated")),
            );
          } else if (state is FamilyDetailsFailure) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text("Failed: ${state.error}")),
            );
          }
        },
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Form(
              key: _formKey,
              child: Column(
                children: [
                  _sectionTitle("Father's Details"),
                  _buildField("Name", _fatherName),
                  _buildField("Occupation", _fatherOccupation),
                  _buildField("Qualification", _fatherQualification),
                  _buildField("Mobile Number", _fatherMobile),
                  _buildField("Email", _fatherEmail),
                  const SizedBox(height: 16),
                  _sectionTitle("Mother's Details"),
                  _buildField("Name", _motherName),
                  _buildField("Occupation", _motherOccupation),
                  _buildField("Qualification", _motherQualification),
                  _buildField("Mobile Number", _motherMobile),
                  _buildField("Email", _motherEmail),
                  const SizedBox(height: 24),
                  BlocBuilder<FamilyDetailsBloc, FamilyDetailsState>(
                    builder: (context, state) => ElevatedButton(
                      onPressed: state is FamilyDetailsSubmitting
                          ? null
                          : () {
                              if (_formKey.currentState?.validate() ?? false) {
                                final model = FamilyDetailsModel(
                                  father: {
                                    'name': _fatherName.text.trim(),
                                    'occupation': _fatherOccupation.text.trim(),
                                    'qualification':
                                        _fatherQualification.text.trim(),
                                    'mobile': _fatherMobile.text.trim(),
                                    'email': _fatherEmail.text.trim(),
                                  },
                                  mother: {
                                    'name': _motherName.text.trim(),
                                    'occupation': _motherOccupation.text.trim(),
                                    'qualification':
                                        _motherQualification.text.trim(),
                                    'mobile': _motherMobile.text.trim(),
                                    'email': _motherEmail.text.trim(),
                                  },
                                );
                                context.read<FamilyDetailsBloc>().add(
                                      SubmitFamilyDetails(model, widget.userId),
                                    );
                              }
                            },
                      child: state is FamilyDetailsSubmitting
                          ? const CircularProgressIndicator()
                          : const Text("Submit"),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildField(String label, TextEditingController controller) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: TextFormField(
          controller: controller,
          decoration: InputDecoration(
            labelText: '$label *',
            border: const OutlineInputBorder(),
          ),
          validator: (value) =>
              value == null || value.trim().isEmpty ? 'Required' : null,
        ),
      );

  Widget _sectionTitle(String title) => Align(
        alignment: Alignment.centerLeft,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Text(
            title,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              color: Colors.blue,
              fontSize: 16,
            ),
          ),
        ),
      );
}
